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
from embed import load_chunks  # noqa: E402
from retrieve import COUNTRY_CONF, fused_scores, keyword_scores, query_scores, rerank, split_scores  # noqa: E402
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
    ap.add_argument("--keyword", nargs="*", default=[], choices=["examples", "terms", "terms_jp", "terms_kr"],
                    help="--fuse에 키워드 검색(pg_bigm)을 더한다. --keyword-only: 키워드 검색 하나만 잰다")
    ap.add_argument("--keyword-only", choices=["examples", "terms_jp", "terms_kr"])
    ap.add_argument("--rerank", type=int, default=0, metavar="N", help="검색 상위 N개를 Ranking API로 다시 줄 세운다 (0 = 안 함)")
    ap.add_argument("--rerank-query", choices=["orig", "rewrite", "both"], default="orig", help="rerank에 넘기는 질의")
    ap.add_argument("--split", action="store_true",
                    help="나라별 분리 검색 (src/retrieve.py COUNTRY_CONF). k는 두 나라 합계이고, 각 나라 근거는 그 나라 상위 k/2 안에서 센다")
    ap.add_argument("--jp-keyword", nargs="*", default=None, help="--split: JP 설정 덮어쓰기")
    ap.add_argument("--jp-rerank", type=int, default=None)
    ap.add_argument("--kr-keyword", nargs="*", default=None)
    ap.add_argument("--kr-rerank", type=int, default=None)
    ap.add_argument("--backend", choices=["db", "npy"], default="db", help="db: pgvector, npy: numpy 전수 비교 (src/retrieve.py)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    chunks = load_chunks()
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    samples = gold_units(golden, chunks)

    client = genai.Client()
    rw = {}
    if args.query != "orig" or args.fuse or args.keyword_only or args.split:
        rw = {r["id"]: r["rewrite"] for r in map(json.loads, (ROOT / f"data/queries/{args.rewrite}.jsonl").read_text().splitlines())}

    def queries(kind):
        return [q if kind == "orig" else rw[sid] if kind == "rewrite" else f"{q}\n{rw[sid]}" for sid, q, _ in samples]

    by_country = None
    if args.split:
        conf = {c: dict(v) for c, v in COUNTRY_CONF.items()}
        for c in ("jp", "kr"):
            if getattr(args, f"{c}_keyword") is not None:
                conf[c.upper()]["keyword"] = tuple(getattr(args, f"{c}_keyword"))
            if getattr(args, f"{c}_rerank") is not None:
                conf[c.upper()]["rerank"] = getattr(args, f"{c}_rerank")
            conf[c.upper()]["rerank_query"] = args.rerank_query if args.rerank_query != "orig" else conf[c.upper()]["rerank_query"]
        print("나라별 설정:", conf)
        by_country = split_scores(client, queries("orig"), queries("rewrite"), args.fuse or "examples_v3", conf)
        scores = by_country["JP"]  # 아래 공통 경로용. 실제 순위는 나라별 행렬에서 다시 구한다
    elif args.keyword_only:
        scores = keyword_scores(queries("orig"), queries("rewrite"), args.keyword_only)
    elif args.fuse:
        scores = fused_scores(client, queries("orig"), queries("rewrite"), args.fuse, args.backend, args.keyword)
    else:
        scores = query_scores(client, queries(args.query), args.examples, args.backend)
    if args.rerank and not args.split:
        if not rw:
            rw = {r["id"]: r["rewrite"] for r in map(json.loads, (ROOT / f"data/queries/{args.rewrite}.jsonl").read_text().splitlines())}
        scores = rerank(queries(args.rerank_query), scores, args.rerank)
    order = np.argsort(-scores, axis=1)  # 전체 순위

    # finding마다 「근거 chunk 중 가장 높은 순위」를 구한다
    rows = []
    for qi, ((sid, _, units), rank, sc) in enumerate(zip(samples, order, scores)):
        pos = {int(ci): (r if np.isfinite(sc[ci]) else float("inf")) for r, ci in enumerate(rank)}
        for country, doc, hits in units:
            if by_country:  # 그 나라 순위표에서 순위를 구한다. 나라별 k/2와 비교하도록 순위를 2배로 센다
                csc = by_country[country][qi]
                pos = {int(ci): (2 * r if np.isfinite(csc[ci]) else float("inf")) for r, ci in enumerate(np.argsort(-csc))}
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
    print(f"\n순위 중앙값 {ranks[n // 2]}, 최악 {ranks[-1]}  (inf = db 검색 후보 밖)")
    miss = sorted((r for r in rows if r["rank"] > args.show), key=lambda r: -r["rank"])
    print(f"\n## top-{args.show}에서 놓친 근거 {len(miss)}개")
    for r in miss:
        print(f"- {r['id']} {r['country']} #{r['rank']} {r['doc']}  (가장 높은 근거 chunk: {r['ref']})")

if __name__ == "__main__":
    main()
