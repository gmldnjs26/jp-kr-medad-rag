"""시술 문구 1건을 Gemini로 판정한다.

- Phase 1 judge(): RAG 없이 core corpus를 통째로 넣는다. corpus는 context cache에 올려 두고 문구마다 짧은 요청만 보낸다.
- Phase 2 judge_rag(): 검색으로 고른 청크 k개만 근거 자료로 넣는다. 판정 기준(policy.md)과 출력 형식은 같다.
"""
import pathlib
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/corpus/core.md"
POLICY = ROOT / "docs/policy.md"


class Finding(BaseModel):
    span: Optional[str] = Field(description="입력 문자열에 글자 그대로 있는 부분 문자열. 누락(C8)이면 null")
    field: Optional[Literal["title", "description"]] = Field(description="span이 있는 필드. 누락이면 null")
    category: Literal["C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    country: Literal["JP", "KR"]
    severity: Literal["violation", "caution"]
    rule: str = Field(description="법령 조문까지. 예: 医療法 第6条の5第1項 / 医療広告等ガイドライン 第3-1 1(1)")
    source_file: str = Field(description="근거 자료 섹션의 source_file 값 그대로")
    reason: str
    suggestion: str = Field(description="원문 언어로 쓴 수정안. 병원이 채울 자리는 [ ]")


class Judgment(BaseModel):
    verdict: Literal["violation", "caution", "ok"]
    findings: list[Finding]


SYSTEM = """あなたは韓国・日本の医療広告規制に詳しい審査担当者です。
韓国の美容医療を日本居住者に紹介する仲介プラットフォーム（運営主体は韓国法人と仮定）の管理画面で、
韓国の医療機関が登録した施術の「タイトル」と「説明」を審査します。

# 判定ルール
以下の「판정 기준 (policy)」が判定基準です。必ずこの基準に従ってください（運営方針、カテゴリ、境界事例を含む）。
- 日本・韓国の両方の基準で審査する。同じ表現が両国に該当する場合は finding を国ごとに分けて書く。
- 該当がなければ verdict=ok、findings=[]。
- verdict は findings の severity の最大値（violation > caution > ok）。
- span は入力に「文字どおり」含まれる最短の部分文字列。情報の欠落（C8）は span=null, field=null。
- rule には下記「根拠資料」に実際に書かれている条文・項目番号を書く。資料にない条文を作らない。
- source_file には根拠資料の各セクション先頭にある source_file の値をそのまま書く。
- suggestion は元の文言と同じ言語で、違反にならない言い換えを書く。

# 판정 기준 (policy)
{policy}

# 根拠資料
以下に法令・ガイドライン・事例集の本文を示す。
"""


def system_text() -> str:
    return SYSTEM.format(policy=POLICY.read_text())


def corpus_content() -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=CORPUS.read_text())])


def make_cache(client: genai.Client, model: str, ttl_seconds: int = 3600) -> Optional[str]:
    """core corpus + 시스템 지시를 명시적 context cache에 올린다.

    모델·리전 조합에 따라 명시적 캐시가 지원되지 않는다 (2026-11 확인: global의 3.x는 403,
    asia-northeast1의 gemini-3.5-flash만 가능). 실패하면 None을 돌려주고, judge()가 자료를 매번 직접 넣는다.
    """
    try:
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name="medad-core-corpus",
                system_instruction=system_text(),
                contents=[corpus_content()],
                ttl=f"{ttl_seconds}s",
            ),
        )
        return cache.name
    except Exception as e:
        print(f"명시적 캐시 사용 불가, 자료를 매 요청에 넣는다: {str(e)[:100]}")
        return None


def judge(client: genai.Client, model: str, cache_name: Optional[str], title: str, description: str):
    """판정 결과(Judgment)와 토큰 사용량을 돌려준다."""
    prompt = f"次の施術を審査してください。\n\n[タイトル]\n{title}\n\n[説明]\n{description}"
    config = dict(temperature=0, response_mime_type="application/json", response_schema=Judgment)
    if cache_name:
        config["cached_content"] = cache_name
        contents = prompt
    else:
        # 자료를 항상 같은 순서로 맨 앞에 둔다 → 암묵적 캐시가 적용될 수 있다
        config["system_instruction"] = system_text()
        contents = [corpus_content(), types.Content(role="user", parts=[types.Part(text=prompt)])]
    resp = client.models.generate_content(
        model=model, contents=contents, config=types.GenerateContentConfig(**config))
    u = resp.usage_metadata
    usage = {
        "prompt": u.prompt_token_count or 0,
        "cached": u.cached_content_token_count or 0,
        "output": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
    }
    return resp.parsed, usage


RAG_NOTE = """
※ 根拠資料は、この施術文句で検索して取り出した抜粋です。資料にない条文は書かないでください。
抜粋に該当する規定がない場合でも、判定基準(policy)に照らして問題があれば finding を書き、rule には判定基準に書かれた条文を書いてください。"""


def context_content(chunks) -> types.Content:
    """검색된 청크를 근거 자료 블록으로. 모델이 rule·source_file을 정확히 쓰도록 ref와 source_file을 앞에 붙인다."""
    body = "\n\n".join(f"## {c['ref']}\nsource_file: {c['source_file']}\n{c['text']}" for c in chunks)
    return types.Content(role="user", parts=[types.Part(text=body)])


def judge_rag(client: genai.Client, model: str, chunks, title: str, description: str):
    """검색된 chunks(dict 목록)만 근거로 판정한다. 판정 결과(Judgment)와 토큰 사용량을 돌려준다."""
    prompt = f"次の施術を審査してください。\n\n[タイトル]\n{title}\n\n[説明]\n{description}"
    config = dict(temperature=0, response_mime_type="application/json", response_schema=Judgment,
                  system_instruction=system_text() + RAG_NOTE)
    contents = [context_content(chunks), types.Content(role="user", parts=[types.Part(text=prompt)])]
    resp = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**config))
    u = resp.usage_metadata
    usage = {
        "prompt": u.prompt_token_count or 0,
        "cached": u.cached_content_token_count or 0,
        "output": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
    }
    return resp.parsed, usage
