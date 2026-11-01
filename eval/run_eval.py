"""평가셋 전체를 판정하고 채점한다 (Phase 1).

사용법:
  .venv/bin/python eval/run_eval.py                       # 기본 모델로 50건
  .venv/bin/python eval/run_eval.py --model gemini-3.1-pro-preview --limit 3
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
from judge import judge, make_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (0 = 전체)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client()
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        golden = golden[: args.limit]

    run = f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{args.model}"
    out_dir = ROOT / "eval/results" / run
    out_dir.mkdir(parents=True)
    print(f"run: {run} / {len(golden)}건 / context cache 생성 중...")
    cache_name = make_cache(client, args.model)

    def one(g):
        try:
            pred, usage = judge(client, args.model, cache_name, g["title"], g["description"])
            return {"id": g["id"], "model": args.model, "prediction": pred.model_dump(), "usage": usage}
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
