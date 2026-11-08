"""Phase 2 검색: 두 검색의 순위를 RRF로 합친다.

- 검색 A: 원문 질의(제목+설명) × max(원문 벡터, 예시 벡터)   (multi-vector, src/augment.py)
- 검색 B: 바꿔 쓴 질의(src/rewrite.py) × 원문 벡터
- 청크 점수 = Σ 1/(K+순위). 점수 척도가 달라도 섞을 수 있고, 양쪽에서 모두 상위인 청크가 높아진다.

검색 평가(eval/retrieval.py --fuse)와 RAG 판정(eval/run_eval.py --rag)이 같은 함수를 쓴다.
"""
import numpy as np

from embed import embed, emb_path

RRF_K = 60


def query_scores(client, queries, examples=None):
    """(질의, chunk) 코사인 점수. examples가 있으면 예시 벡터와 비교한 점수와 큰 쪽을 쓴다."""
    qv = embed(client, queries, "RETRIEVAL_QUERY")
    sc = qv @ np.load(emb_path()).T
    if examples:
        sc = np.maximum(sc, qv @ np.load(emb_path(kind=examples)).T)  # 예시가 없는 chunk의 행은 0 벡터
    return sc


def rrf(*score_mats):
    """점수 행렬 여러 개를 순위로 합친다."""
    total = np.zeros_like(score_mats[0])
    for sc in score_mats:
        rank = np.argsort(np.argsort(-sc, axis=1), axis=1)  # 0부터 시작하는 순위
        total += 1.0 / (RRF_K + rank + 1)
    return total


def fused_scores(client, queries, rewrites, examples):
    """queries: 제목+설명, rewrites: 같은 순서의 바꿔 쓴 질의"""
    return rrf(query_scores(client, queries, examples), query_scores(client, rewrites))
