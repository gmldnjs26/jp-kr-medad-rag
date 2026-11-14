"""청크와 임베딩을 로컬 Postgres(pgvector)에 적재한다 (Phase 2).

임베딩 API는 부르지 않는다. src/embed.py가 만든 .npy를 그대로 넣는다.
전체를 지우고 다시 넣으므로 여러 번 실행해도 결과가 같다.

사용법:
  docker compose up -d                      # DB 기동 (db/README.md)
  .venv/bin/python src/load_db.py           # 원문 벡터 + examples_v3 예시 벡터
  .venv/bin/python src/load_db.py --examples examples_v2
"""
import argparse
import json
import os

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from embed import CHUNKS, DIM, MODEL, emb_path, load_chunks

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://medad:medad@localhost:5432/medad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="examples_v3", help="예시 벡터 이름 (data/chunks/<NAME>_<model>_<dim>.npy)")
    args = ap.parse_args()

    chunks = load_chunks()
    orig = np.load(emb_path())
    ex = np.load(emb_path(kind=args.examples))
    assert len(chunks) == len(orig) == len(ex), "청크 수와 벡터 행 수가 다르다. embed.py를 다시 돌린다"
    model = f"{MODEL}_{DIM}"
    ex_text = [json.loads(l) for l in (CHUNKS.parent / f"{args.examples}.jsonl").read_text().splitlines() if l.strip()]
    assert {e["id"] for e in ex_text} == {c["id"] for c in chunks}, "예시 파일의 id가 chunks.jsonl과 다르다"  # 행 순서는 달라도 된다

    vec_rows = []
    for c, o, e in zip(chunks, orig, ex):
        vec_rows.append((c["id"], "orig", model, o))
        if np.any(e):  # 예시가 없는 청크는 0 벡터로 저장돼 있다. 코사인 거리가 NaN이 되므로 행을 만들지 않는다
            vec_rows.append((c["id"], "example", model, e))

    with psycopg.connect(DB_URL) as conn:  # with 블록이 끝나면 commit, 예외가 나면 rollback
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks CASCADE")  # chunk_vectors도 함께 비운다
            cur.executemany(
                "INSERT INTO chunks (id, country, doc, ref, source_file, page, pages, text)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [(c["id"], c["country"], c["doc"], c["ref"], c["source_file"],
                  c.get("page"), c.get("pages"), c["text"]) for c in chunks])
            cur.executemany(
                "INSERT INTO chunk_vectors (chunk_id, kind, model, embedding) VALUES (%s, %s, %s, %s)",
                vec_rows)
            cur.executemany(
                "INSERT INTO chunk_examples (chunk_id, idx, text) VALUES (%s, %s, %s)",
                [(e["id"], i, t) for e in ex_text for i, t in enumerate(e.get("examples") or [])])
            cur.execute("SELECT kind, count(*) FROM chunk_vectors GROUP BY kind ORDER BY kind")
            vecs = dict(cur.fetchall())
            cur.execute("SELECT count(*) FROM chunk_examples")
            print(f"chunks {len(chunks)}, vectors {vecs}, example sentences {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
