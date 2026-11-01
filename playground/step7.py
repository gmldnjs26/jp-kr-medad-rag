"""Step 7: 근거 자료 전체(core corpus 18만 토큰) 넣기

목적: 필요한 자료를 전부 넣으면 Step 6에서 놓친 것을 잡는지, 대신 무엇을 잃는지 본다. (= Phase 1 방식)
Step 6과 다른 점:
- law를 조문 2개 → data/corpus/core.md 전체로 바꿈. 나머지(지시·기준·샘플 3건)는 그대로
"""

import json
import sys
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, "src")
from build_corpus import kr_articles, jp_articles

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

# 判定基準（会社の決定事項）
- 韓国の医師を「医師」「専門医」と表記することは許容する。判定しない。
- 「ダウンタイムが少ない」は比較対象が不明確なため caution。具体的な期間を書けば ok。
"""

# ---- 여기부터 바뀐 부분: 여러 샘플을 돌며 정답과 나란히 출력 ----
gold = {json.loads(l)["id"]: json.loads(l) for l in open("eval/golden.jsonl")}

for ID in ["g005", "g008", "g040"]:
    g = gold[ID]
    text = f"次の施術広告を審査してください。\n\n[タイトル]{g['title']}\n[説明]{g['description']}"
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
    j = resp.parsed
    print(f"\n== {ID} 정답 {g['verdict']} / 예측 {j.verdict}")
    for f in g["findings"]:
        print("  정답", f["country"], f["span"], "|", f["rule"])
    for f in j.findings:
        print("  예측", f.country, f.span, "|", f.rule)

# == g005 정답 violation / 예측 violation
#   정답 JP ボツラックス | 医薬品医療機器等法 第68条 / 医療広告等ガイドライン 第3-1 1(5)・1(8)イ①
#   정답 KR ボツラックス | 약사법 제68조제6항제1호
#   정답 JP 1回の注射で小顔に | 医療法 第6条の5第2項第2号 / 医療広告等ガイドライン 第3-1 1(3)
#   정답 JP None | 医療法施行規則 第1条の9の2 / 医療広告等ガイドライン 第3-3 2 ③④
#   예측 JP ボツラックス | 医薬品医療機器等法 第六十八条
#   예측 JP 1回の注射で小顔に | 医療法 第六条の五第二項第二号
#   예측 KR ボツラックス | 약사법 제68조제6항제1호
#   예측 KR 1回の注射で小顔に | 의료법 제56조제2항제2호
#
# == g008 정답 violation / 예측 violation
#   정답 JP 人気K-POPアイドルも通う | 医療法 第6条の5第2項第1号 / 医療広告等ガイドライン 第3-1 1(2)
#   정답 KR 人気K-POPアイドルも通う | 의료법 제56조제2항제14호
#   정답 JP あのアイドルが通っていることで話題 | 医療法 第6条の5第2項第1号 / 医療広告等ガイドライン 第3-1 1(2)
#   예측 JP 人気K-POPアイドルも通う | 医療法 第6条の5第2項第1号
#   예측 JP あのアイドルが通っていることで話題 | 医療法 第6条の5第2項第1号
#   예측 KR 人気K-POPアイドルも通う | 의료법 제56조제2항제2호
#   예측 KR あのアイドルが通っていることで話題 | 의료법 제56조제2항제2호
#
# == g040 정답 violation / 예측 violation
#   정답 JP 첫 방문 고객 20% 할인 | 医療広告等ガイドライン 第3-1 1(8)ア①
#   정답 JP 150,000원 → 120,000원 | 医療広告等ガイドライン 第3-1 1(8)ア①
#   정답 JP None | 医療法施行規則 第1条の9の2 / 医療広告等ガイドライン 第3-3 2 ③④
#   정답 KR None | 의료법 제56조제2항제7호
#   예측 JP 첫 방문 고객 20% 할인 (11월 한정) | 医療法 第6条の5第3項
#   예측 KR 비급여 레이저 토닝 | 의료법 제56조제2항제7호
#   예측 KR 첫 방문 고객 20% 할인 (11월 한정) | 의료법 제27조제3항

# ===== 정리 =====
# Step 6 대비 달라진 점:
# - g005: 상품명 「ボツラックス」를 JP 薬機法 68条, KR 약사법 68조⑥1호로 정확히 잡았다.
# - g008: 일본 조문이 1号(비교우량)로 바뀌어 정답과 일치. 가이드라인의 유명인 예시를 읽었다.
# 한계:
# - 한국 조문이 다시 56조②2호로 퇴행(g005·g008). 56조가 18만 토큰 속에 묻히자 잘못된 기억이 돌아왔다(lost in the middle).
# - g040: 가이드라인 1(8)ア①이 자료에 있는데도 찾지 못하고 여전히 第3項. 「150,000원 → 120,000원」도 놓침.
# - C8 span이 null이 아니라 「비급여 레이저 토닝」. null 규칙을 아직 안 줬다.
# - 느리다. 캐시는 비용만 줄이고 18만 토큰을 읽는 시간은 그대로다. 관리화면 응답으로 쓰기 어렵다.
# → 적게 넣으면 빠뜨리고(Step 6), 많이 넣으면 묻히고 느려진다(Step 7). 샘플마다 필요한 조문만 골라 넣자 = RAG(Phase 2).
