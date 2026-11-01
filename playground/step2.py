"""Step 2: 출력 형식 고정 (response_schema)

목적: 자유 서술 대신 Pydantic 스키마로 JSON을 강제하면 결과가 어떻게 달라지는지 본다.
Step 1과 다른 점:
- response_schema=Judgment, temperature=0 추가
- 질문 문장이 「日本や韓国の…違反する表現はありますか？」→「審査してください」로 바뀜 (한 번에 두 가지를 바꾼 실수)
"""

from typing import Literal, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(".env")
client = genai.Client()

class Finding(BaseModel):
    span: Optional[str]          # 문제 문구 (입력에 있는 그대로)
    country: Literal["JP", "KR"]
    severity: Literal["violation", "caution"]
    rule: str                    # 조문 번호
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
# JP violation 痛みゼロ | 医療広告ガイドライン（誇大広告・効果の保証の禁止）
# JP violation 絶対に腫れない | 医療広告ガイドライン（虚偽・誇大広告、絶対的表現の禁止）
# JP violation 絶対安全な施術です | 医療広告ガイドライン（虚偽・誇大広告、安全性の保証の禁止）
# JP violation ダウンタイムはありません | 医療広告ガイドライン（虚偽・誇大広告の禁止）
# ---
# output token: 546
# thinking token: 283

# ===== 정리 =====
# Step 1 대비 달라진 점:
# - resp.parsed로 바로 파이썬 객체를 받는다. 출력 546 / thinking 283 토큰으로 절반 이하.
# 한계:
# - 한국 finding이 사라졌다. 질문에서 '日本や韓国'이 빠지자 일본 법만 봤다. 스키마에 KR 자리가 있어도 지시가 없으면 채우지 않는다.
# - rule이 「医療広告ガイドライン（誇大広告…）」처럼 조문 번호 없이 흐려졌다.
# → 스키마는 결과의 '모양'만 정하고, '내용의 기준'은 지시로 줘야 한다.
