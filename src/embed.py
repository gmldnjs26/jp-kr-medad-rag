"""chunks.jsonl을 임베딩해서 로컬 파일에 저장하고, 질의 벡터로 top-k를 찾는다 (Phase 2).

DB 없이 numpy 행렬로 검색한다. 591 chunks 규모에서는 전수 비교(코사인)로 충분하다.

사용법:
  .venv/bin/python src/embed.py              # 원문 → data/chunks/embeddings_<model>_<dim>.npy
  .venv/bin/python src/embed.py --examples [NAME]   # data/chunks/NAME.jsonl(기본 examples)의 예시 → NAME_<model>_<dim>.npy
결과 파일의 i번째 행은 chunks.jsonl의 i번째 줄과 대응한다.
"""
import json
import pathlib
import sys

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data/chunks/chunks.jsonl"
MODEL = "gemini-embedding-001"  # gemini-embedding-2는 리스트 입력을 벡터 1개로 합쳐 버려 배치가 안 된다 (2026-11 확인)
DIM = 768
BATCH = 50


def emb_path(model=MODEL, dim=DIM, kind="chunks"):
    """kind: chunks(원문) / 예시 파일 이름(examples, examples_v2 …: src/augment.py의 예시를 " / "로 이은 텍스트)"""
    return ROOT / f"data/chunks/{'embeddings' if kind == 'chunks' else kind}_{model}_{dim}.npy"


def load_chunks():
    return [json.loads(l) for l in CHUNKS.read_text().splitlines() if l.strip()]


def embed(client, texts, task_type, model=MODEL, dim=DIM):
    """texts를 BATCH개씩 임베딩해 L2 정규화한 (n, dim) 행렬을 돌려준다."""
    out = []
    for i in range(0, len(texts), BATCH):
        r = client.models.embed_content(
            model=model, contents=texts[i:i + BATCH],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim))
        assert len(r.embeddings) == len(texts[i:i + BATCH]), "임베딩 개수가 입력과 다르다"
        out.extend(e.values for e in r.embeddings)
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
    m = np.array(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def search(doc_matrix, query_vecs, k):
    """코사인 유사도 상위 k개 chunk의 행 번호. (질의 수, k)"""
    return np.argsort(-(query_vecs @ doc_matrix.T), axis=1)[:, :k]


def main():
    """인자 없음: 원문 임베딩. --examples: 예시 임베딩 (chunks.jsonl과 같은 행 순서, 예시 없는 chunk는 0 벡터)"""
    load_dotenv(ROOT / ".env")
    client = genai.Client()
    chunks = load_chunks()
    if "--examples" in sys.argv:
        rest = sys.argv[sys.argv.index("--examples") + 1:]
        name = rest[0] if rest else "examples"
        ex = {r["id"]: r["examples"] for r in map(json.loads, (ROOT / f"data/chunks/{name}.jsonl").read_text().splitlines())}
        rows = [i for i, c in enumerate(chunks) if ex.get(c["id"])]
        m = np.zeros((len(chunks), DIM), dtype=np.float32)
        m[rows] = embed(client, [" / ".join(ex[chunks[i]["id"]]) for i in rows], "RETRIEVAL_DOCUMENT")
        path = emb_path(kind=name)
    else:
        m = embed(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
        path = emb_path()
    np.save(path, m)
    print(f"{len(chunks)} chunks × {DIM}차원 → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
