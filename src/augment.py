"""청크마다 「여기에 해당하는 광고 문구 예시」를 붙인다 (Phase 2, multi-vector 검색용).

- 법령·JP 가이드라인 청크: LLM이 예시를 새로 만든다 (generate)
- 事例解説書·복지부 가이드라인 청크: 페이지에 실린 광고 예시를 그대로 뽑는다 (extract)

예시는 검색용 미끼일 뿐이고, 판정 LLM에는 넘기지 않는다.

v2 (2026-11-03, 예시 검수 결과 반영):
  1. 광고와 무관한 조문(벌칙, 助産所, 행정처분, 심의 절차, 등록 요건 등)은 SKIP 규칙으로 비운다
  2. 같은 상위 조문의 형제 호·항 본문을 함께 보여 주고 「この条文だけ」に該当する例を要求한다
  3. 시술명 예시 목록을 빼고, 시술명은 「〇〇」로 伏せる (시술 어휘가 모든 청크에 공통 성분이 되는 것을 막는다)
  4. 括弧書きの解説・※注記를 금지한다
  extract 행은 v1 결과를 그대로 복사한다 (--reuse).

사용법:
  .venv/bin/python src/augment.py --out examples_v2 --reuse examples [--limit N]
결과: data/chunks/<out>.jsonl  {"id", "mode", "examples": [...], "usage": {input, output, thinking}}  (usage는 2026-11-03 v2c부터)  (이미 있는 id는 건너뛴다)
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from embed import load_chunks  # noqa: E402

MODEL = "gemini-3.8-flash"
EXTRACT_DOCS = {  # 페이지 단위 문서: 실린 예시를 그대로 뽑는다 (2026-11-07 자료 재선정)
    "医療広告ガイドラインQ&A", "不当な価格表示についての考え方", "ステマ告示運用基準",
    "KHIDI 외국인환자유치 온라인 의료광고 가이드라인", "의협 의료광고 사전자율심의기준",
    "식약처 의약품 광고 가이드라인", "식약처 의료기기법 위반광고 해설", "복지부 유형별 의료광고 사례 및 체크리스트"}

# v1 (examples.jsonl). 기록용으로 남긴다
GENERATE_V1 = """次は医療広告規制の条文(またはガイドラインの一節)です。
この条文に違反する、または該当する美容医療の広告文句の例を
日本語で5つ、韓国語で5つ書いてください。
- 実際に病院が書きそうな短いタイトル・説明文の口調で
- 施術名(二重、ボトックス、糸リフト等)を混ぜて多様に
- 広告に関係しない条文(名称、手続き等)なら「なし」とだけ答える

[条文]
{text}"""

GENERATE = """次は医療広告規制の条文(またはガイドラインの一節)です。
この条文「だけ」に該当し、下に挙げた隣の条文には該当しない美容医療の広告文句の例を
日本語で5つ、韓国語で5つ書いてください。
- 実際に病院が書きそうな短いタイトル・説明文の口調で
- 広告に実際に載る文言だけを書く。括弧書きの解説、※注記、「〜という広告」のような説明は書かない
- 施術名は具体名を書かず「〇〇」と伏せる(例: 〇〇施術、〇〇注射)
- 隣の条文と共通する表現(効果の保証、副作用ゼロ等)は、この条文の中心的な内容でない限り使わない
- 広告に関係しない条文(名称、手続き等)なら「なし」とだけ答える

[条文]
{text}
{siblings}"""

# ablation용 변형 (2026-11-03). v2a: 공통 표현 금지 규칙 삭제 / v2b: 거기에 「この条文だけ」·형제 표시도 삭제
RULE_COMMON = "- 隣の条文と共通する表現(効果の保証、副作用ゼロ等)は、この条文の中心的な内容でない限り使わない\n"
GENERATE_V2A = GENERATE.replace(RULE_COMMON, "")
GENERATE_V2B = (GENERATE_V2A
                .replace("この条文「だけ」に該当し、下に挙げた隣の条文には該当しない美容医療の広告文句の例を",
                         "この条文に違反する、または該当する美容医療の広告文句の例を")
                .replace("{text}\n{siblings}", "{text}"))
# v2c: v2b에서 시술명 「〇〇」 가리기도 삭제 (시술명이 광고 질의와의 유사도를 만든다는 가설 검증)
GENERATE_V2C = GENERATE_V2B.replace("- 施術名は具体名を書かず「〇〇」と伏せる(例: 〇〇施術、〇〇注射)\n", "")
PROMPTS = {"v2": GENERATE, "v2a": GENERATE_V2A, "v2b": GENERATE_V2B, "v2c": GENERATE_V2C}

# 광고 문구와 무관한 조문. 예시를 만들지 않는다 (ref 앞부분 일치)
SKIP = re.compile(r"^(医療法 第6条の[78]|医療法 第(28|29|87|89)条|医療法施行規則 第1条の10|"
                  r"医療広告等ガイドライン 第[156](\s|$)|의료법 제27조제[145]항|의료법 제57조제([2-9]|1[01])항|"
                  r"의료법 시행령 제23조제3항|의료법 시행령 제24조|의료해외진출법 제6조)")


def parent(ref):
    """형제를 묶는 상위 ref. 호가 있으면 호를, 없으면 항을, GL이면 마지막 (n)을 뗀다."""
    for pat in (r"(第\d+号|제\d+호)$", r"(第\d+項|제\d+항)$", r"\(\d+\)[^ ]*$"):
        if re.search(pat, ref):
            return re.sub(pat, "", ref)
    return None


def article(ref):
    """조 단위 ref. 医療法 第6条の5第2項第2号 → 医療法 第6条の5, GL 第3-1 1(3) → GL 第3-1 1"""
    m = re.match(r"(.*?(第\d+条(の\d+)*|제\d+조(의\d+)?))", ref)
    return m.group(1) if m else re.sub(r"\(\d+\)[^ ]*$", "", ref)


def body(chunk, n=150):
    """형제 조문을 구별하는 본문. 법령이고 「헤더 / 항 머리말 / 호」 3줄 이상이면 마지막 줄(호), 아니면 헤더 뒤 앞부분."""
    lines = [l for l in chunk["text"].split("\n") if l.strip()]
    is_law = chunk["doc"] != "医療広告等ガイドライン"  # GL은 줄이 문장 중간에서 끊겨 있어 앞부분을 쓴다
    return (lines[-1] if is_law and len(lines) >= 3 else " ".join(lines[1:]))[:n]


EXTRACT = """次は医療広告の事例集・ガイドラインの1ページです。
このページに実際に載っている「広告文句の例」(違反例・不適切な表現の例)を、
本文に書かれているとおりに抜き出してください。言い換えや新しい例の作成はしないでください。
- 1つの例は1つの文字列にする
- 広告文句の例が載っていないページなら「なし」とだけ答える

[ページ]
{text}"""


def ask(client, chunk, siblings, variant="v2"):
    if chunk["doc"] in EXTRACT_DOCS:
        mode, prompt = "extract", EXTRACT.format(text=chunk["text"])
    elif SKIP.match(chunk["ref"]):
        return {"id": chunk["id"], "mode": "skip", "examples": []}
    else:
        sib = "".join(f"- {c['ref']}: {body(c)}\n" for c in siblings)
        mode, prompt = "generate", PROMPTS[variant].format(text=chunk["text"], siblings=f"\n[隣の条文]\n{sib}" if sib else "")
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7, response_mime_type="application/json", response_schema=list[str]))
    ex = [s.strip() for s in (resp.parsed or []) if s and s.strip() and s.strip() != "なし"]
    u = resp.usage_metadata
    usage = {"input": u.prompt_token_count or 0, "output": u.candidates_token_count or 0,
             "thinking": u.thoughts_token_count or 0}  # thinking도 출력 단가로 청구된다
    return {"id": chunk["id"], "mode": mode, "examples": ex, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="examples_v2")
    ap.add_argument("--reuse", default=None, help="이 파일의 extract 행을 그대로 복사한다")
    ap.add_argument("--variant", default="v2", choices=list(PROMPTS))
    ap.add_argument("--only", default=None, help="ref가 이 정규식에 맞는 chunk만 새로 만든다")
    ap.add_argument("--base", default=None, help="--only 밖의 chunk는 이 파일의 행을 그대로 복사한다")
    args = ap.parse_args()
    OUT = ROOT / f"data/chunks/{args.out}.jsonl"

    load_dotenv(ROOT / ".env")
    client = genai.Client(http_options=types.HttpOptions(timeout=120_000))  # 응답 없는 요청이 작업 전체를 멈추지 않게 (ms)
    chunks = load_chunks()
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    if args.reuse:
        with OUT.open("a") as f:
            for r in map(json.loads, (ROOT / f"data/chunks/{args.reuse}.jsonl").read_text().splitlines()):
                if r["mode"] == "extract" and r["id"] not in done:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done.add(r["id"])
    if args.base:
        only = re.compile(args.only)
        by_id = {c["id"]: c for c in chunks}
        with OUT.open("a") as f:
            for r in map(json.loads, (ROOT / f"data/chunks/{args.base}.jsonl").read_text().splitlines()):
                if r["id"] not in done and not only.match(by_id[r["id"]]["ref"]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done.add(r["id"])

    def siblings(c):
        """같은 상위 조문의 형제를 먼저, 그다음 같은 조(GL은 같은 절)의 다른 항·호. 최대 16개."""
        p, art = parent(c["ref"]), article(c["ref"])
        near = [s for s in chunks if p and s["doc"] == c["doc"] and parent(s["ref"]) == p and s["ref"] != c["ref"]]
        far = [s for s in chunks if s["doc"] == c["doc"] and article(s["ref"]) == art
               and s["ref"] != c["ref"] and s not in near]
        return (near + far)[:16]

    todo = [c for c in chunks if c["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)}개 처리 (이미 {len(done)}개 있음)")

    total = {"calls": 0, "input": 0, "output": 0, "thinking": 0}
    with OUT.open("a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask, client, c, siblings(c), args.variant): c["id"] for c in todo}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            try:
                r = fut.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if "usage" in r:
                    total["calls"] += 1
                    for k, v in r["usage"].items():
                        total[k] += v
            except Exception as e:
                print(f"실패 {futs[fut]}: {str(e)[:120]}")
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"LLM 호출 {total['calls']}회 / 토큰 입력 {total['input']:,}, 출력 {total['output']:,}, thinking {total['thinking']:,}")


if __name__ == "__main__":
    main()
