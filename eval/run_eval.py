"""평가셋 전체를 판정하고 채점한다 (Phase 1).

사용법:
  .venv/bin/python eval/run_eval.py                       # 기본 모델로 50건
  .venv/bin/python eval/run_eval.py --model gemini-3.1-pro-preview --limit 3
  .venv/bin/python eval/run_eval.py --rag 10      # Phase 2: RRF 검색 상위 10개 청크만 근거로 판정
  .venv/bin/python eval/run_eval.py --corpus data/corpus/core_v2.md   # 재선정 자료 전체를 통째로 (long-context 비교)
결과: eval/results/<날짜시각>_<모델>/predictions.jsonl, summary.md
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import pathlib
import subprocess
import sys

from dotenv import load_dotenv
from google import genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import judge as judge_mod  # noqa: E402
from judge import judge, judge_rag, make_cache  # noqa: E402
from embed import load_chunks  # noqa: E402
from retrieve import fused_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (0 = 전체)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--corpus", default="data/corpus/core.md",
                    help="Phase 1(--rag 0)에서 통째로 넣는 자료. core_v2.md = 재선정 자료 13개 (src/chunk.py 결과)")
    ap.add_argument("--rag", type=int, default=0, metavar="K", help="Phase 2: 검색 상위 K개 청크만 근거로 판정 (0 = Phase 1)")
    ap.add_argument("--examples", default="examples_v3", help="--rag 검색에 쓰는 예시 벡터")
    ap.add_argument("--rewrite", default="rewrite_v1", help="--rag 검색에 쓰는 바꿔 쓴 질의 (data/queries/<name>.jsonl)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    judge_mod.CORPUS = ROOT / args.corpus  # judge()·make_cache()가 이 파일을 읽는다
    client = genai.Client()
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        golden = golden[: args.limit]

    run = f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{args.model}" + (f"_rag{args.rag}" if args.rag else f"_{pathlib.Path(args.corpus).stem}")
    out_dir = ROOT / "eval/results" / run
    out_dir.mkdir(parents=True)
    retrieved = {}
    if args.rag:
        # 검색은 한 번에 (임베딩 배치). 바꿔 쓴 질의는 미리 만든 것을 쓴다 (src/rewrite.py)
        chunks = load_chunks()
        rw = {r["id"]: r["rewrite"] for r in map(json.loads, (ROOT / f"data/queries/{args.rewrite}.jsonl").read_text().splitlines())}
        scores = fused_scores(client, [f"{g['title']}\n{g['description']}" for g in golden], [rw[g["id"]] for g in golden],
                              args.examples)
        for g, row in zip(golden, scores):
            retrieved[g["id"]] = [chunks[int(i)] for i in row.argsort()[::-1][:args.rag]]
        print(f"run: {run} / {len(golden)}건 / RAG top-{args.rag}")
        cache_name = None
    else:
        print(f"run: {run} / {len(golden)}건 / context cache 생성 중...")
        cache_name = make_cache(client, args.model)

    def one(g):
        try:
            if args.rag:
                got = retrieved[g["id"]]
                pred, usage = judge_rag(client, args.model, got, g["title"], g["description"])
                extra = {"retrieved": [c["id"] for c in got]}
            else:
                pred, usage = judge(client, args.model, cache_name, g["title"], g["description"])
                extra = {}
            return {"id": g["id"], "model": args.model, "prediction": pred.model_dump(), "usage": usage, **extra}
        except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다
            return {"id": g["id"], "model": args.model, "error": str(e)[:500]}

    results = []
    try:
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for r in ex.map(one, golden):
                mark = "ERR" if "error" in r else r["prediction"]["verdict"]
                print(f"  {r['id']}: {mark}")
                results.append(r)
    finally:
        if cache_name:
            client.caches.delete(name=cache_name)

    pred_path = out_dir / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n오류 {len(errors)}건: " + "; ".join(f"{r['id']}: {r['error'][:120]}" for r in errors))
    subprocess.run([sys.executable, str(ROOT / "eval/score.py"), str(pred_path)], check=True)


if __name__ == "__main__":
    main()
