"""Phase 2 검색: 두 검색의 순위를 RRF로 합친다.

- 검색 A: 원문 질의(제목+설명) × max(원문 벡터, 예시 벡터)   (multi-vector, src/augment.py)
- 검색 B: 바꿔 쓴 질의(src/rewrite.py) × 원문 벡터
- 청크 점수 = Σ 1/(K+순위). 점수 척도가 달라도 섞을 수 있고, 양쪽에서 모두 상위인 청크가 높아진다.

백엔드
- db (기본): 로컬 Postgres + pgvector (src/load_db.py로 적재). 벡터마다 가까운 CANDIDATES개만 받는다.
  후보 밖의 청크는 점수가 -inf이고, RRF에서 0점이다.
- npy: data/chunks/*.npy를 numpy로 전부 비교한다. db와 결과가 같은지 확인할 때 쓴다.
db의 예시 벡터는 load_db.py가 넣은 것 하나뿐이다. examples 인자는 npy에서만 파일 이름으로 쓰고, db에서는 「예시 벡터를 쓰는가」만 뜻한다.

검색 평가(eval/retrieval.py)와 RAG 판정(eval/run_eval.py --rag)이 같은 함수를 쓴다.
"""
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embed import DIM, MODEL, embed, emb_path, load_chunks

RRF_K = 60
CANDIDATES = 200  # db: 벡터 종류마다 가져오는 후보 수. 100이면 RRF 꼬리가 잘려 recall@10이 74→72% (2026-11-08)
DB_URL = os.getenv("DATABASE_URL", "postgresql://medad:medad@localhost:5432/medad")

# <=> 는 pgvector의 코사인 거리(1 - 코사인 유사도). 인덱스(HNSW 등)를 쓰려면 ORDER BY에 이 연산자를 그대로 둔다
NEAREST_SQL = """
SELECT chunk_id, 1 - (embedding <=> %(q)s) AS sim
FROM chunk_vectors
WHERE model = %(model)s AND kind = %(kind)s
ORDER BY embedding <=> %(q)s
LIMIT %(n)s
"""


def _db_scores(qv, kinds):
    """질의마다 kind별로 가까운 CANDIDATES개를 받아, 청크마다 가장 높은 유사도를 행렬에 넣는다. 받지 못한 청크는 -inf"""
    index = {c["id"]: i for i, c in enumerate(load_chunks())}
    sc = np.full((len(qv), len(index)), -np.inf)
    with psycopg.connect(os.getenv("DATABASE_URL", DB_URL)) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for row, q in zip(sc, qv):
                for kind in kinds:
                    cur.execute(NEAREST_SQL, {"q": q, "model": f"{MODEL}_{DIM}", "kind": kind, "n": CANDIDATES})
                    for chunk_id, sim in cur.fetchall():
                        i = index[chunk_id]
                        row[i] = max(row[i], sim)
    return sc


def query_scores(client, queries, examples=None, backend="db"):
    """(질의, chunk) 코사인 점수. examples가 있으면 예시 벡터와 비교한 점수와 큰 쪽을 쓴다."""
    qv = embed(client, queries, "RETRIEVAL_QUERY")
    if backend == "db":
        return _db_scores(qv, ["orig", "example"] if examples else ["orig"])
    sc = qv @ np.load(emb_path()).T
    if examples:
        sc = np.maximum(sc, qv @ np.load(emb_path(kind=examples)).T)  # 예시가 없는 chunk의 행은 0 벡터
    return sc


def rrf(*score_mats):
    """점수 행렬 여러 개를 순위로 합친다. -inf(검색 후보 밖)는 0점."""
    total = np.zeros_like(score_mats[0])
    for sc in score_mats:
        rank = np.argsort(np.argsort(-sc, axis=1), axis=1)  # 0부터 시작하는 순위
        total += np.where(np.isfinite(sc), 1.0 / (RRF_K + rank + 1), 0.0)
    return np.where(total > 0, total, -np.inf)  # 어느 검색에도 들지 못한 청크는 -inf로 둔다


def fused_scores(client, queries, rewrites, examples, backend="db"):
    """queries: 제목+설명, rewrites: 같은 순서의 바꿔 쓴 질의"""
    return rrf(query_scores(client, queries, examples, backend), query_scores(client, rewrites, backend=backend))
