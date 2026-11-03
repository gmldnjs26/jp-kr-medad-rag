"""Step 5: 판정 기준(운영 방침) 넣기

목적: 법 조문만으로는 판단할 수 없는 '서비스 운영 방침(가정)'이 결과를 바꾸는지 본다.
Step 4와 다른 점:
- 샘플을 golden.jsonl에서 ID로 읽음. g004(정답 ok: 韓国形成外科専門医), g037(정답 caution: ダウンタイムが少ない)
- 기준 없이 2건 실행 → SYSTEM에 「判定基準」 두 줄을 추가하고 2건 다시 실행 (아래 결과 순서: g004, g037, g004, g037)
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
    span: Optional[str]          # 문제 문구 (입력에 있는 그대로)
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
""" + law

SYSTEM += """

# 判定基準（運営方針・仮定）
- 韓国の医師を「医師」「専門医」と表記することは許容する。判定しない。
- 「ダウンタイムが少ない」は比較対象が不明確なため caution。具体的な期間を書けば ok。
"""

ID = "g037"   # g037로도 실행
g = next(json.loads(l) for l in open("eval/golden.jsonl") if json.loads(l)["id"] == ID)
text = f"次の施術広告を審査してください。\n\n[タイトル]{g['title']}\n[説明]{g['description']}"
print("정답:", g["verdict"], [(f["country"], f["span"]) for f in g["findings"]])

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
j = resp.parsed                  # 이미 Judgment 객체
print(j.verdict)
for f in j.findings:
    print(f.country, f.severity, f.span, "|", f.rule)
print("---")
print(resp.usage_metadata.candidates_token_count, resp.usage_metadata.thoughts_token_count)


# $ .venv/bin/python playground/step5.py
# 정답: ok []
# violation
# JP violation 韓国形成外科専門医 | 医療法 第六条の五第三項
# ---
# 168 1561
# $ .venv/bin/python playground/step5.py
# 정답: caution [('JP', 'ダウンタイムが少ない')]
# caution
# JP caution ダウンタイムが少ない | 医療法 第六条の五第２項第二号
# KR caution ダウンタイムが少ない | 의료법 제56조제2항제8호
# ---
# 333 784
# $ .venv/bin/python playground/step5.py
# 정답: ok []
# ok
# ---
# 9 387
# $ .venv/bin/python playground/step5.py
# 정답: caution [('JP', 'ダウンタイムが少ない')]
# caution
# JP caution ダウンタイムが少ない | 医療法 第六条の五第2項第二号
# KR caution ダウンタイムが少ない | 의료법 제56조제2항제8호
# ---
# 184 643

# ===== 정리 =====
# Step 4 대비 달라진 점:
# - g004: 기준 없이 violation(6条の5第3項, 広告可能事項 외) → 기준 넣고 ok. 법만 보면 모델이 맞고, 운영 방침이 결과를 뒤집었다.
#   thinking도 1,561 → 387 토큰으로 줄었다.
# - g037: 기준 유무와 관계없이 caution. 이 경계 사례 규칙은 이 샘플에서는 효과가 없었다(ablation).
# 한계:
# - g037에서 KR 8호 caution도 냈는데 정답엔 JP만 있다. 정답 쪽 보완 후보.
# - 조문 표기가 「第２項」「第2項」「第二号」로 제각각. 스키마 지시나 채점기 정규화가 필요하다.
# → 라벨링 가이드(판정 기준)를 프롬프트에 넣는 이유: 모델이 법을 몰라서가 아니라 서비스의 운영 방침을 몰라서.
