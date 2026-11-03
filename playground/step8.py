"""Step 8: 응답 시간 측정 + 채점기(score.py)가 무엇을 맞았다고 세는지 보기

목적:
- 18만 토큰을 넣었을 때 1건 응답에 몇 초 걸리는지, 시간이 입력·출력 중 무엇에 비례하는지 본다.
- 실제 예측을 eval/score.py의 span_match / articles로 채점해, 채점기가 너무 엄격하거나 후한 곳을 찾는다.
Step 7과 다른 점:
- 프롬프트(core.md 전체 + 판정 기준)는 그대로. 샘플은 g005, g008 두 건
- generate_content 앞뒤로 time.perf_counter() 측정, usage_metadata(입력·캐시·출력·thinking) 출력
- 예측을 정답 finding별로 [찾음]/[놓침]/[제외] + 조문 일치 여부로 출력
"""

import json
import sys
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, "src")
sys.path.insert(0, "eval")
from score import span_match, articles

load_dotenv(".env")
client = genai.Client()


class Finding(BaseModel):
    span: Optional[str]
    country: Literal["JP", "KR"]
    severity: Literal["violation", "caution"]
    rule: str = Field(description="法令名と条文番号まで。例: 医療法 第6条の5第1項 / 의료법 제56조제2항제3호")
    reason: str
    suggestion: str


class Judgment(BaseModel):
    verdict: Literal["violation", "caution", "ok"]
    findings: list[Finding]


law = open("data/corpus/core.md").read()

SYSTEM = """あなたは韓国・日本の医療広告規制の審査担当者です。
日本と韓国の両方の基準で審査し、同じ表現が両国に該当する場合は finding を国ごとに分けて書いてください。
rule には下記の条文に実際に書かれている項・号の番号を書き、条文にないものを作らないでください。
span は入力に文字どおり含まれる最短の部分文字列にしてください。

# 根拠条文
""" + law + """

# 判定基準（運営方針・仮定）
- 韓国の医師を「医師」「専門医」と表記することは許容する。判定しない。
- 「ダウンタイムが少ない」は比較対象が不明確なため caution。具体的な期間を書けば ok。
"""

# ---- 여기부터: API 호출 시간을 재고, 결과를 score.py 함수로 바로 채점 ----
gold = {json.loads(l)["id"]: json.loads(l) for l in open("eval/golden.jsonl")}

for ID in ["g005", "g008"]:
    g = gold[ID]
    text = f"次の施術広告を審査してください。\n\n[タイトル]{g['title']}\n[説明]{g['description']}"

    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Judgment,
        ),
    )
    elapsed = time.perf_counter() - t0

    u = resp.usage_metadata
    print(f"\n== {ID}  {elapsed:.1f}초 | 입력 {u.prompt_token_count:,} (캐시 {u.cached_content_token_count or 0:,})"
          f" / 출력 {u.candidates_token_count} / thinking {u.thoughts_token_count}")

    preds = [(f.country, f.span, f.rule) for f in resp.parsed.findings]
    for c, s, r in preds:
        print(f"  예측 {c} {s} | {r}")

    for gf in g["findings"]:
        if gf["span"] is None:
            print(f"  [제외] {gf['country']} C8 (span 없음 → span 재현율 계산에서 빠짐)")
            continue
        hit = next((p for p in preds if p[0] == gf["country"] and span_match(gf["span"], p[1])), None)
        if not hit:
            print(f"  [놓침] {gf['country']} {gf['span']}")
            continue
        g_art, p_art = articles(gf["rule"]), articles(hit[2])
        print(f"  [찾음] {gf['country']} {gf['span']}")
        print(f"         정답 조문 {g_art} / 예측 조문 {p_art} → {'일치' if g_art & p_art else '불일치'}")


# == g005  26.9초 | 입력 179,752 (캐시 0) / 출력 748 / thinking 1075
#   예측 JP ボツラックス | 医薬品医療機器等法 第六十八条
#   예측 JP 1回の注射で小顔に | 医療法 第六条の五第二項第二号
#   예측 KR ボツラックス | 약사법 제68조제6항제1호
#   예측 KR 1回の注射で小顔に | 의료법 제56조제2항제2호
#   [찾음] JP ボツラックス
#          정답 조문 {'第3-11(5)', '第68条'} / 예측 조문 set() → 불일치
#   [찾음] KR ボツラックス
#          정답 조문 {'제68조'} / 예측 조문 {'제68조'} → 일치
#   [찾음] JP 1回の注射で小顔に
#          정답 조문 {'第3-11(3)', '第6条の5'} / 예측 조문 set() → 불일치
#   [제외] JP C8 (span 없음 → span 재현율 계산에서 빠짐)
#
# == g008  15.2초 | 입력 179,738 (캐시 0) / 출력 522 / thinking 710
#   예측 JP 人気K-POPアイドルも通う | 医療法 第6条の5第2項第1号
#   예측 JP あのアイドルが通っていることで話題 | 医療法 第6条の5第2項第1号
#   예측 KR 人気K-POPアイドルも通う | 의료법 제56조제2항제2호
#   예측 KR あのアイドルが通っていることで話題 | 의료법 제56조제2항제2호
#   [찾음] JP 人気K-POPアイドルも通う
#          정답 조문 {'第3-11(2)', '第6条の5'} / 예측 조문 {'第6条の5'} → 일치
#   [찾음] KR 人気K-POPアイドルも通う
#          정답 조문 {'제56조'} / 예측 조문 {'제56조'} → 일치
#   [찾음] JP あのアイドルが通っていることで話題
#          정답 조문 {'第3-11(2)', '第6条の5'} / 예측 조문 {'第6条の5'} → 일치
# (위 조문 키는 고치기 전 채점기 출력)

# ===== 정리 =====
# 시간:
# - 입력은 둘 다 약 18만 토큰인데 26.9초 vs 15.2초. 출력+thinking(1,823 vs 1,232 토큰)에 비례했다.
#   → 시간 대부분은 18만 토큰을 읽는 게 아니라 '생성'에 쓰인다. 출력을 줄이는 게 속도에 직결된다.
# - 두 번째 건도 캐시 0. 암묵적 캐시는 보장되지 않는다(Phase 1은 50건 중 대부분 적중).
# - Phase 1의 1건 약 48초는 라벨링 가이드 전문으로 출력이 더 길었고(약 3,300 토큰) 5건 동시 실행이었기 때문.
# 채점기 한계 (이 결과로 발견):
# 1. 한자 숫자(第六十八条)를 못 읽어 set() → 맞는 답 2개를 불일치로 셈 (너무 엄격)
# 2. 조(条) 단위까지만 비교 → g008 KR 정답 14호 / 예측 2호를 일치로 셈 (너무 후함)
# 3. 정답에 없는 예측은 채점 안 함 → g005 KR 小顔 2호(오답)가 안 보임 (정밀도 지표 없음)
# 4. C8(span 없음)은 span 재현율에서 빠지고 verdict에만 간접 반영
# → 이후 score.py 수정: NFKC + 한자 숫자 변환, 항·호까지 키 추출(条6の5-2-1 / 조56-2-14 / GL3-1-1-5),
#   「·제8호」「・1(8)」 나열 표기 처리, 한쪽이 더 세밀하면 상위 번호 일치로 판정(article_match).
#   Phase 1 조문 일치율: 고치기 전 83/91 (91%) → 고친 후 81/91 (89%). 3·4번은 아직 남음.
