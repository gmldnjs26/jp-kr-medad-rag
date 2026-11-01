"""Step 3: 최소한의 지시 추가 (system_instruction + Field description)

목적: 스키마는 '모양'만 정한다. 무엇을 써야 하는지(양국 판정, 조문 번호)는 지시로 줘야 하는지 확인한다.
Step 2와 다른 점:
- system_instruction: 심사 담당자 역할 + 일본·한국 양국을 나라별로 판정
- rule 필드에 description: 법령명과 조문 번호까지, 예시 포함
"""

from typing import Literal, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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

text = """次の施術広告を審査してください。

[タイトル]【痛みゼロ】絶対に腫れない二重埋没法
[説明]当院の埋没法は絶対安全な施術です。ダウンタイムはありません。"""

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=text,
    config=types.GenerateContentConfig(
        system_instruction="""あなたは韓国・日本の医療広告規制の審査担当者です。
日本と韓国の両方の基準で審査し、同じ表現が両国に該当する場合は finding を国ごとに分けて書いてください。""",
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

# violation
# JP violation 【痛みゼロ】 | 医療法 第6条の5第1項
# KR violation 【痛みゼロ】 | 의료법 제56조제2항제3호
# JP violation 絶対に腫れない | 医療法 第6条の5第1項
# KR violation 絶対に腫れない | 의료법 제56조제2항제3호
# JP violation 絶対安全な施術です | 医療法 第6条の5第2項第2号
# KR violation 絶対安全な施術です | 의료법 제56조제2항제2호
# JP violation ダウンタイムはありません | 医療法 第6条の5第2項第2号
# KR violation ダウンタイムはありません | 의료법 제56조제2항제3호
# ---
# 1007 357

# ===== 정리 =====
# Step 2 대비 달라진 점:
# - 한국 finding이 돌아왔고, rule에 조문 번호가 들어갔다. (출력 1,007 토큰: 나라별로 나눠서 늘어남)
# 한계:
# - 일본 1項(허위)과 2項2号(과장)를 섞어 씀: 痛みゼロ→1項(정답 2項2号), 絶対安全→2項2号(정답 1項).
# - 한국 絶対安全을 또 56조②2호(치료경험담)로 인용. 지시로는 안 고쳐지는 '잘못된 기억'.
# - span에 괄호까지 포함(【痛みゼロ】). 자르는 규칙이 없어서.
# - score.py는 조(条) 단위까지만 비교하므로 1項/2項 혼동을 틀렸다고 세지 못한다.
