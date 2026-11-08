"""검색 재현율: 정답 finding의 근거(evidence)를 top-k chunk 안에 찾아오는가 (Phase 2).

질의는 샘플 1건의 「タイトル + 説明」 그대로다 (RAG 판정 때 쓰는 질의와 같다).
채점 단위는 finding 하나다 (2026-11-07~). finding.evidence의 위치 중 하나라도 top-k 안이면 성공이다.
- evidence {doc, page, quote}: 그 문서·페이지의 chunk 중 quote 앞부분이 들어 있는 chunk (없으면 그 페이지 chunk 전부)
- evidence {doc, ref}: 그 문서에서 ref가 같거나 article_match로 맞는 chunk
샘플 안에서 근거 chunk 집합이 같은 finding은 하나로 센다. evidence가 빈 finding(검색 대상 밖 자료)은 뺀다.

사용법:
  .venv/bin/python eval/retrieval.py [--k 5 10 20] [--show 10] [--examples [NAME]]
--examples NAME: 원문 벡터에 예시 벡터(NAME, 기본 examples)를 더해 chunk마다 높은 쪽 점수를 쓴다 (multi-vector).
"""
import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np
from dotenv import load_dotenv
from google import genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from embed import embed, emb_path, load_chunks  # noqa: E402
from score import article_match  # noqa: E402


def norm(t):
    return re.sub(r"\s+", "", t)


def evidence_chunks(e, chunks):
    """evidence 하나 → 대응 chunk 행 번호 목록"""
    if "page" in e:
        # 절 단위 chunk는 여러 페이지에 걸친다 (pages). 페이지 단위 chunk는 page 하나
        page = [i for i, c in enumerate(chunks) if c["doc"] == e["doc"] and e["page"] in c.get("pages", [c.get("page")])]
        quoted = [i for i in page if norm(e.get("quote", ""))[:20] in norm(chunks[i]["text"])]
        return quoted or page
    return [i for i, c in enumerate(chunks)
            if c["doc"] == e["doc"] and "page" not in c and (c["ref"] == e["ref"] or article_match(e["ref"], c["ref"]))]


def gold_units(golden, chunks):
    """[(샘플 id, 질의, [(country, 주 근거 doc, 근거 chunk 집합)])]"""
    out = []
    for g in golden:
        units, seen = [], set()
        for f in g["findings"]:
            hits = frozenset(i for e in f.get("evidence", []) for i in evidence_chunks(e, chunks))
            if hits and hits not in seen:
                seen.add(hits)
                units.append((f["country"], f["evidence"][0]["doc"], hits))
        out.append((g["id"], f"{g['title']}\n{g['description']}", units))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--show", type=int, default=10, help="이 k에서 놓친 인용을 나열")
    ap.add_argument("--examples", nargs="?", const="examples", default=None, help="예시 벡터도 함께 쓴다 (multi-vector)")
    ap.add_argument("--query", choices=["orig", "rewrite", "both"], default="orig",
                    help="orig: 제목+설명 / rewrite: src/rewrite.py의 쟁점 목록 / both: 둘을 이어 붙임")
    ap.add_argument("--rewrite", default="rewrite_v1", help="data/queries/<name>.jsonl")
    ap.add_argument("--fuse", default=None, metavar="EXAMPLES",
                    help="RRF: 「원문 질의 + 예시 벡터(EXAMPLES)」와 「바꿔 쓴 질의 + 원문 벡터」의 순위를 합친다")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    chunks = load_chunks()
    docs = np.load(emb_path())
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    samples = gold_units(golden, chunks)

    client = genai.Client()
    rw = {}
    if args.query != "orig" or args.fuse:
        rw = {r["id"]: r["rewrite"] for r in map(json.loads, (ROOT / f"data/queries/{args.rewrite}.jsonl").read_text().splitlines())}

    def scores_for(kind, examples):
        """kind: orig/rewrite/both 질의로 (질의, chunk) 점수. examples가 있으면 예시 벡터와 max."""
        qs = [q if kind == "orig" else rw[sid] if kind == "rewrite" else f"{q}\n{rw[sid]}" for sid, q, _ in samples]
        qv = embed(client, qs, "RETRIEVAL_QUERY")
        sc = qv @ docs.T
        if examples:
            ex = np.load(emb_path(kind=examples))  # 예시가 없는 chunk의 행은 0 벡터 → 점수 0
            sc = np.maximum(sc, qv @ ex.T)
        return sc

    if args.fuse:
        # RRF: 두 검색의 순위로 1/(K+순위)를 더한다. 점수 척도가 달라도 섞을 수 있다
        K = 60
        a = scores_for("orig", args.fuse)    # 원문 질의 + 예시 벡터
        b = scores_for("rewrite", None)      # 바꿔 쓴 질의 + 원문 벡터
        scores = np.zeros_like(a)
        for sc in (a, b):
            rank = np.argsort(np.argsort(-sc, axis=1), axis=1)  # 0부터 시작하는 순위
            scores += 1.0 / (K + rank + 1)
    else:
        scores = scores_for(args.query, args.examples)
    order = np.argsort(-scores, axis=1)  # 전체 순위

    # finding마다 「근거 chunk 중 가장 높은 순위」를 구한다
    rows = []
    for (sid, _, units), rank in zip(samples, order):
        pos = {int(ci): r for r, ci in enumerate(rank)}
        for country, doc, hits in units:
            best = min(hits, key=pos.get)
            rows.append(dict(id=sid, set="기존 g001-g050" if int(sid[1:]) <= 50 else "신규 g051-",
                             country=country, doc=doc, rank=pos[best] + 1, ref=chunks[best]["ref"]))

    n = len(rows)
    skipped = sum(not f.get("evidence") for g in golden for f in g["findings"])
    print(f"채점 단위(finding) {n}개 (evidence가 없는 finding {skipped}개는 제외)\n")
    print("| 구분 | n | " + " | ".join(f"recall@{k}" for k in args.k) + " |")
    print("|---|---|" + "---|" * len(args.k))
    groups = [("전체", rows)] + [(x, [r for r in rows if r["set"] == x]) for x in ("기존 g001-g050", "신규 g051-")]
    groups += [(c, [r for r in rows if r["country"] == c]) for c in ("JP", "KR")]
    groups += [(d, [r for r in rows if r["doc"] == d]) for d in sorted({r["doc"] for r in rows})]
    for name, rs in groups:
        cells = [f"{sum(r['rank'] <= k for r in rs) / len(rs):.0%}" for k in args.k]
        print(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")

    ranks = sorted(r["rank"] for r in rows)
    print(f"\n순위 중앙값 {ranks[n // 2]}, 최악 {ranks[-1]}")
    miss = sorted((r for r in rows if r["rank"] > args.show), key=lambda r: -r["rank"])
    print(f"\n## top-{args.show}에서 놓친 근거 {len(miss)}개")
    for r in miss:
        print(f"- {r['id']} {r['country']} #{r['rank']} {r['doc']}  (가장 높은 근거 chunk: {r['ref']})")

if __name__ == "__main__":
    main()
