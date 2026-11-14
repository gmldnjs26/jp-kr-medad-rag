"""Phase 2 검색: 여러 검색의 순위를 RRF로 합친다.

- 검색 A: 원문 질의(제목+설명) × max(원문 벡터, 예시 벡터)   (multi-vector, src/augment.py)
- 검색 B: 바꿔 쓴 질의(src/rewrite.py) × 원문 벡터
- 키워드 검색 (하이브리드, pg_bigm. keyword 인자로 고른다)
  - examples: 원문 질의 × 청크의 광고 예시 문장. 2글자 단위 유사도(bigm_similarity)의 청크별 최댓값
  - terms: 바꿔 쓴 질의의 쟁점 용어 × 청크 원문. 들어 있는 용어의 IDF 합
- 청크 점수 = Σ 1/(K+순위). 점수 척도가 달라도 섞을 수 있고, 양쪽에서 모두 상위인 청크가 높아진다.

백엔드
- db (기본): 로컬 Postgres + pgvector (src/load_db.py로 적재). 벡터마다 가까운 CANDIDATES개만 받는다.
  후보 밖의 청크는 점수가 -inf이고, RRF에서 0점이다.
- npy: data/chunks/*.npy를 numpy로 전부 비교한다. db와 결과가 같은지 확인할 때 쓴다.
db의 예시 벡터는 load_db.py가 넣은 것 하나뿐이다. examples 인자는 npy에서만 파일 이름으로 쓰고, db에서는 「예시 벡터를 쓰는가」만 뜻한다.

나라별 분리 검색 (2026-11-14~): JP 청크와 KR 청크를 따로 순위 매기고 각각 상위 k/2개를 뽑는다(split_top).
한 순위표에서 10자리를 나눠 가지면, 한 나라를 올리는 방법이 다른 나라 자리를 뺏기 때문이다. 나라별 설정은 COUNTRY_CONF.

Rerank (2단계): 위 검색의 상위 N개를 Vertex AI Ranking API(Discovery Engine)로 다시 줄 세운다.
질의와 청크를 한 쌍으로 모델에 넣어 관련도를 매기므로 임베딩 비교보다 정밀하지만 느리다. 레코드당 1024토큰까지만 읽는다.

검색 평가(eval/retrieval.py)와 RAG 판정(eval/run_eval.py --rag)이 같은 함수를 쓴다.
"""
import concurrent.futures as cf
import os
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embed import DIM, MODEL, embed, emb_path, load_chunks

RRF_K = 60
CANDIDATES = 200  # db: 벡터 종류마다 가져오는 후보 수. 100이면 RRF 꼬리가 잘려 recall@10이 74→72% (2026-11-08)
DB_URL = os.getenv("DATABASE_URL", "postgresql://medad:medad@localhost:5432/medad")

# <=> 는 pgvector의 코사인 거리(1 - 코사인 유사도). 인덱스(HNSW 등)를 쓰려면 ORDER BY에 이 연산자를 그대로 둔다
# country가 NULL이면 두 나라 전부에서 찾는다
NEAREST_SQL = """
SELECT v.chunk_id, 1 - (v.embedding <=> %(q)s) AS sim
FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id
WHERE v.model = %(model)s AND v.kind = %(kind)s AND (%(country)s::text IS NULL OR c.country = %(country)s)
ORDER BY v.embedding <=> %(q)s
LIMIT %(n)s
"""


# 키워드 검색 examples: 예시 문장마다 2글자 단위 유사도를 재고, 청크별 최댓값을 점수로 쓴다
EXAMPLE_KEYWORD_SQL = """
SELECT chunk_id, max(bigm_similarity(%(q)s, text)) AS sim
FROM chunk_examples
GROUP BY chunk_id
HAVING max(bigm_similarity(%(q)s, text)) > 0
ORDER BY sim DESC
LIMIT %(n)s
"""

# 키워드 검색 terms: 용어가 원문에 들어 있으면 그 용어의 IDF(ln(전체 청크 수 / 그 용어가 든 청크 수))를 더한다.
# 드문 용어일수록 무겁다. 「중요」「정보」처럼 흔한 단어는 0에 가깝다.
# 한 나라의 청크 안에서만 순위를 낸다. 섞으면 한국어 단어가 더 잘게 나뉘어 많이 맞으므로 KR 청크가 JP 청크를 밀어낸다
TERM_KEYWORD_SQL = """
WITH t AS (SELECT DISTINCT unnest(%(terms)s::text[]) AS term),
c AS (SELECT id, regexp_replace(text, '\\s', '', 'g') AS text FROM chunks WHERE country = %(country)s),  -- PDF 줄바꿈이 단어를 끊으므로 공백을 지운다
hit AS (SELECT t.term, c.id FROM t JOIN c ON c.text LIKE '%%' || t.term || '%%'),
idf AS (SELECT term, ln((SELECT count(*) FROM c)::float / count(*)) AS w FROM hit GROUP BY term)
SELECT hit.id, sum(idf.w) AS score
FROM hit JOIN idf USING (term)
GROUP BY hit.id
HAVING sum(idf.w) > 0
ORDER BY score DESC
LIMIT %(n)s
"""


def rewrite_terms(rewrite):
    """바꿔 쓴 질의 「* 誇大な表現（誇大広告） / 부작용 등 중요 정보 누락」에서 용어를 뽑는다 → 誇大な表現, 誇大広告, 부작용, 중요, 정보, 누락"""
    terms = set()
    for line in rewrite.splitlines():
        line = re.sub(r"^\s*(?:[*\-・]|\d+[.)])\s*", "", line)
        for t in re.split(r"[\s/（）()、,・]+", line):  # 한국어는 띄어쓰기, 일본어는 ・ 와 괄호로 나눈다
            if len(t) >= 2:
                terms.add(t)
    return sorted(terms)


def _sql_scores(sql, params):
    """params(질의마다 SQL 인자 dict)로 sql을 돌려 (질의, chunk) 점수 행렬을 만든다. 결과에 없는 청크는 -inf"""
    index = {c["id"]: i for i, c in enumerate(load_chunks())}
    sc = np.full((len(params), len(index)), -np.inf)
    with psycopg.connect(os.getenv("DATABASE_URL", DB_URL)) as conn:
        with conn.cursor() as cur:
            for row, p in zip(sc, params):
                cur.execute(sql, {**p, "n": CANDIDATES})
                for chunk_id, v in cur.fetchall():
                    row[index[chunk_id]] = v
    return sc


def keyword_scores(queries, rewrites, kind):
    if kind == "examples":
        return _sql_scores(EXAMPLE_KEYWORD_SQL, [{"q": q} for q in queries])
    if kind in ("terms_jp", "terms_kr"):
        country, script = ("JP", r"[\u3040-\u30ff\u4e00-\u9fff]") if kind == "terms_jp" else ("KR", r"[\uac00-\ud7a3]")
        return _sql_scores(TERM_KEYWORD_SQL, [{"country": country, "terms": [t for t in rewrite_terms(r) if re.search(script, t)]}
                                              for r in rewrites])
    raise ValueError(kind)


KEYWORD_KINDS = {"examples": ["examples"], "terms": ["terms_jp", "terms_kr"], "terms_jp": ["terms_jp"], "terms_kr": ["terms_kr"]}  # terms는 나라별 목록 두 개로 RRF에 들어간다


def _mask(sc, country):
    """country 밖의 청크를 -inf로 지운다 (country가 None이면 그대로)"""
    if country:
        sc = sc.copy()
        sc[:, [c["country"] != country for c in load_chunks()]] = -np.inf
    return sc


_EMB_CACHE = {}


def _embed_queries(client, queries):
    """나라별 검색에서 같은 질의를 두 번 임베딩하지 않도록 캐시한다"""
    key = tuple(queries)
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = embed(client, queries, "RETRIEVAL_QUERY")
    return _EMB_CACHE[key]


def _db_scores(qv, kinds, country=None):
    """질의마다 kind별로 가까운 CANDIDATES개를 받아, 청크마다 가장 높은 유사도를 행렬에 넣는다. 받지 못한 청크는 -inf"""
    index = {c["id"]: i for i, c in enumerate(load_chunks())}
    sc = np.full((len(qv), len(index)), -np.inf)
    with psycopg.connect(os.getenv("DATABASE_URL", DB_URL)) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for row, q in zip(sc, qv):
                for kind in kinds:
                    cur.execute(NEAREST_SQL, {"q": q, "model": f"{MODEL}_{DIM}", "kind": kind, "country": country, "n": CANDIDATES})
                    for chunk_id, sim in cur.fetchall():
                        i = index[chunk_id]
                        row[i] = max(row[i], sim)
    return sc


def query_scores(client, queries, examples=None, backend="db", country=None):
    """(질의, chunk) 코사인 점수. examples가 있으면 예시 벡터와 비교한 점수와 큰 쪽을 쓴다. country: 그 나라 청크만"""
    qv = _embed_queries(client, queries)
    if backend == "db":
        return _db_scores(qv, ["orig", "example"] if examples else ["orig"], country)
    sc = qv @ np.load(emb_path()).T
    if examples:
        sc = np.maximum(sc, qv @ np.load(emb_path(kind=examples)).T)  # 예시가 없는 chunk의 행은 0 벡터
    return _mask(sc, country)


def rrf(*score_mats):
    """점수 행렬 여러 개를 순위로 합친다. -inf(검색 후보 밖)는 0점."""
    total = np.zeros_like(score_mats[0])
    for sc in score_mats:
        rank = np.argsort(np.argsort(-sc, axis=1), axis=1)  # 0부터 시작하는 순위
        total += np.where(np.isfinite(sc), 1.0 / (RRF_K + rank + 1), 0.0)
    return np.where(total > 0, total, -np.inf)  # 어느 검색에도 들지 못한 청크는 -inf로 둔다


def fused_scores(client, queries, rewrites, examples, backend="db", keyword=(), country=None):
    """queries: 제목+설명, rewrites: 같은 순서의 바꿔 쓴 질의, keyword: 함께 합칠 키워드 검색 (examples, terms)"""
    mats = [query_scores(client, queries, examples, backend, country), query_scores(client, rewrites, backend=backend, country=country)]
    mats += [_mask(keyword_scores(queries, rewrites, k), country) for kw in keyword for k in KEYWORD_KINDS[kw]]
    return rrf(*mats)


RERANK_MODEL = "semantic-ranker-default@latest"


def rerank(queries, scores, n):
    """질의마다 scores 상위 n개를 Ranking API로 다시 줄 세운다. 그 n개가 맨 위로 가고, 나머지는 원래 순서를 지킨다."""
    from google.cloud import discoveryengine_v1 as de

    chunks = load_chunks()
    client = de.RankServiceClient(transport="rest")  # gRPC는 이 환경에서 DNS 조회가 간헐적으로 실패했다
    config = client.ranking_config_path(os.environ["GOOGLE_CLOUD_PROJECT"], "global", "default_ranking_config")

    def one(q, row):
        top = [int(i) for i in np.argsort(-row)[:n] if np.isfinite(row[i])]
        res = client.rank(request=de.RankRequest(
            ranking_config=config, model=RERANK_MODEL, query=q, ignore_record_details_in_response=True,
            records=[de.RankingRecord(id=str(i), title=chunks[i]["ref"], content=chunks[i]["text"]) for i in top]))
        return {int(r.id): r.score for r in res.records}

    out = scores.copy()
    with cf.ThreadPoolExecutor(5) as ex:
        for row, ranked in zip(out, ex.map(one, queries, scores)):
            for i, s in ranked.items():
                row[i] = 10 + s  # RRF 점수(최대 0.05 정도)보다 항상 위
    return out


# 나라별 검색 설정. keyword: fused_scores의 keyword, rerank: 상위 몇 개를 rerank할지(0 = 안 함), rerank_query: orig | rewrite | both
COUNTRY_CONF = {
    "JP": {"keyword": (), "rerank": 0, "rerank_query": "rewrite"},
    "KR": {"keyword": (), "rerank": 50, "rerank_query": "rewrite"},
}


def split_scores(client, queries, rewrites, examples, conf=None):
    """나라마다 점수 행렬을 따로 만든다. 다른 나라 청크는 -inf. → {"JP": 행렬, "KR": 행렬}"""
    out = {}
    for country, c in (conf or COUNTRY_CONF).items():
        sc = fused_scores(client, queries, rewrites, examples, keyword=c["keyword"], country=country)
        if c["rerank"]:
            rq = {"orig": queries, "rewrite": rewrites, "both": [f"{q}\n{r}" for q, r in zip(queries, rewrites)]}[c["rerank_query"]]
            sc = rerank(rq, sc, c["rerank"])
        out[country] = sc
    return out


def split_top(scores_by_country, k):
    """질의마다 나라별 상위 k/2개씩 이어 붙인 청크 번호 목록 (JP 먼저)"""
    half = k // 2
    n = len(next(iter(scores_by_country.values())))
    return [[int(i) for sc in scores_by_country.values() for i in np.argsort(-sc[q])[:half]] for q in range(n)]
