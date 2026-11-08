"""광고 문구를 「법적 쟁점의 말」로 바꿔 쓴 검색 질의를 만든다 (Phase 2, query rewriting).

광고 문구(「痛みゼロ」)와 법 문장(「誇大な広告」)은 쓰는 말이 달라 임베딩이 가깝지 않다.
검색 전에 LLM이 쟁점을 법률 용어로 나열하게 해서, 질의를 법 문장 쪽으로 옮긴다.
빠진 정보(비용·리스크 등)도 쟁점으로 쓰게 해서, 원문에 없는 누락(C8)도 질의에 나타나게 한다.

판정 기준(policy.md)이나 정답은 넣지 않는다. 넣으면 검색 평가가 정답을 미리 아는 셈이 된다.

사용법:
  .venv/bin/python src/rewrite.py [--name rewrite_v1]
결과: data/queries/<name>.jsonl  {"id", "rewrite", "usage"}  (이미 있는 id는 건너뛴다)
"""
import argparse
import concurrent.futures as cf
import json
import pathlib

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "gemini-3.8-flash"

PROMPT = """次は美容医療クリニックが登録した施術の広告文です。
この広告文が日本と韓国の医療広告規制で問題になりうる論点を列挙してください。
- 各論点を1行で、規制で使われる法律用語で書く(例: 効果の保証、誇大な表現、体験談、費用の記載なし)
- 広告文にある表現の問題だけでなく、書かれていない情報(費用、リスク・副作用、未承認医薬品である旨など)も論点にする
- 各行は日本語の後に韓国語を併記する(例: 効果の保証 / 효과 보장)
- 条文番号は書かない
- 問題がなさそうなら「なし」とだけ答える

[タイトル]
{title}

[説明]
{description}"""


def rewrite(client, g):
    resp = client.models.generate_content(
        model=MODEL, contents=PROMPT.format(title=g["title"], description=g["description"]),
        config=types.GenerateContentConfig(temperature=0))
    u = resp.usage_metadata
    usage = {"input": u.prompt_token_count or 0, "output": u.candidates_token_count or 0,
             "thinking": u.thoughts_token_count or 0}
    return {"id": g["id"], "rewrite": (resp.text or "").strip(), "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rewrite_v1")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    out = ROOT / f"data/queries/{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    load_dotenv(ROOT / ".env")
    client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [g for g in golden if g["id"] not in done]
    print(f"{len(todo)}건 처리 (이미 {len(done)}건 있음)")

    total = {"input": 0, "output": 0, "thinking": 0}
    with out.open("a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        for fut in cf.as_completed([ex.submit(rewrite, client, g) for g in todo]):
            try:
                r = fut.result()
            except Exception as e:
                print(f"실패: {str(e)[:120]}")
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            for k, v in r["usage"].items():
                total[k] += v
    print(f"토큰 입력 {total['input']:,}, 출력 {total['output']:,}, thinking {total['thinking']:,}")


if __name__ == "__main__":
    main()
