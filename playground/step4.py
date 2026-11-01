"""Step 4: 근거 조문 2개만 넣기

목적: 모델의 기억 대신 조문 원문을 주면 조문 번호 인용이 정확해지는지 본다. (= RAG의 기본 아이디어를 손으로 해 본 것)
Step 3과 다른 점:
- 医療法 6条の5, 의료법 56조 원문(약 2천 토큰)을 system_instruction에 추가
- 「条文にないものを作らない」「span은 최단 부분 문자열」 지시 추가
- 준비: build_corpus.jp_articles가 항·호 번호(２, 一, 二)를 빼먹던 버그를 고침
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

import sys
sys.path.insert(0, "src")
from build_corpus import kr_articles, jp_articles

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

# violation
# JP violation 痛みゼロ | 医療法 第六条の五第二項第二号
# KR violation 痛みゼロ | 의료법 제56조제2항제8호
# JP violation 絶対に腫れない | 医療法 第六条の五第一項
# KR violation 絶対に腫れない | 의료법 제56조제2항제3호
# JP violation 絶対安全 | 医療法 第六条の五第一項
# KR violation 絶対安全 | 의료법 제56조제2항제3호
# JP violation ダウンタイムはありません | 医療法 第六条の五第二項第二号
# KR violation ダウンタイムはありません | 의료법 제56조제2항제8호
# ---
# 1001 549

# ===== 정리 =====
# Step 3 대비 달라진 점:
# - 일본 1項/2項2号가 정답과 일치, 한국 2호 오류가 사라지고 3호(거짓)·8호(과장)를 골랐다.
# - span이 최단으로 잘렸다(痛みゼロ, 絶対安全).
# - 입력이 약 2천 토큰 늘었고 출력은 비슷.
# 한계:
# - 「ダウンタイムはありません」의 리스크 누락(C8) 측면을 놓쳤다. 법에 없는 '우리의 해석 규칙'이라 조문으로는 못 준다.
# - 조문 번호가 「第六条の五第二項」처럼 한자 숫자로 나온다. score.py의 정규식은 아라비아 숫자만 잡아서 맞아도 불일치로 센다.
