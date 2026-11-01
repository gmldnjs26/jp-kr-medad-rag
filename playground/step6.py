"""Step 6: 여러 건 돌려서 어디서 깨지는지 보기

목적: 프롬프트에 없는 자료가 필요한 샘플에서 모델이 어떻게 행동하는지 본다.
Step 5와 다른 점:
- 프롬프트(조문 2개 + 판정 기준)는 그대로, 샘플만 3건으로 바꿔 반복문으로 실행하고 정답과 나란히 출력
- g005(상품명 → 薬機法), g008(유명인 → ガイドライン 1(2)), g040(할인 전후 가격 → ガイドライン 1(8)ア①)
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


law = (
    "## [JP] 医療法\n" + jp_articles("jp/医療法_323AC0000000205_施行20261001.xml", ["6_5"]) +
    "\n\n## [KR] 의료법\n" + kr_articles("kr/의료법_20261002.txt", ["56조"])
)

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
#   예측 JP 1回の注射で小顔に | 医療法第六条の五第二項第二号
#   예측 KR 1回の注射で小顔に | 의료법 제56조제2항제8호

# == g008 정답 violation / 예측 violation
#   정답 JP 人気K-POPアイドルも通う | 医療法 第6条の5第2項第1号 / 医療広告等ガイドライン 第3-1 1(2)
#   정답 KR 人気K-POPアイドルも通う | 의료법 제56조제2항제14호
#   정답 JP あのアイドルが通っていることで話題 | 医療法 第6条の5第2項第1号 / 医療広告等ガイドライン 第3-1 1(2)
#   예측 JP 人気K-POPアイドルも通う | 医療法 第六条の五第2項第二号
#   예측 KR 人気K-POPアイドルも通う | 의료법 제56조제2항제14호
#   예측 JP あのアイドルが通っていることで話題 | 医療法 第六条の五第2項第二号
#   예측 KR あのアイドルが通っていることで話題 | 의료법 제56조제2항제14호

# == g040 정답 violation / 예측 violation
#   정답 JP 첫 방문 고객 20% 할인 | 医療広告等ガイドライン 第3-1 1(8)ア①
#   정답 JP 150,000원 → 120,000원 | 医療広告等ガイドライン 第3-1 1(8)ア①
#   정답 JP None | 医療法施行規則 第1条の9の2 / 医療広告等ガイドライン 第3-3 2 ③④
#   정답 KR None | 의료법 제56조제2항제7호
#   예측 KR 첫 방문 고객 20% 할인 (11월 한정) | 의료법 제56조제2항제13호
#   예측 JP 첫 방문 고객 20% 할인 (11월 한정) | 医療法 第六条の五第三項

# ===== 정리 =====
# Step 5 대비 달라진 점: 3건 모두 verdict는 맞았다. 하지만 근거는 프롬프트에 없는 부분이 전부 틀렸다.
# 한계:
# - g005: 상품명 「ボツラックス」를 놓침(薬機法이 없어서). 과장 표현 덕에 verdict만 맞았다.
# - g008: 일본 2項2号로 인용(정답 1号 비교우량). 유명인=비교우량이라는 해석은 가이드라인 예시에만 있다.
# - g040: 일본 근거를 받은 조문 중 가까운 6条の5第3項에 끼워 맞춤. 한국은 할인 조건이 명시됐는데 13호 위반으로 잡음(경계 규칙 없음).
# - C8(비용·리스크 누락)은 전부 놓침.
# → verdict만 보면 3/3이라 문제가 안 보인다. span·조문까지 채점하는 이유.
# → 모델은 다른 법을 지어내지 않고 받은 조문 안에서 끼워 맞췄다. 어떤 자료를 넣느냐가 결과를 좌우한다.
