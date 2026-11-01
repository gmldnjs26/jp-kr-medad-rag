"""golden.jsonl을 사람이 읽기 쉬운 Markdown(eval/golden.md)으로 변환한다.

사용법: python3 eval/render.py   (프로젝트 루트에서 실행)
원본은 golden.jsonl이다. golden.md는 보기용이니 직접 고치지 말고, jsonl을 고친 뒤 다시 실행한다.
"""
import collections
import html
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "eval/golden.jsonl"
OUT = ROOT / "eval/golden.md"

BADGE = {"violation": "🔴 위반", "caution": "🟡 주의", "ok": "🟢 OK"}
CATEGORY = {
    "C1": "과장·최상급·보장", "C2": "체험담·비포애프터", "C3": "외국 의사 표기",
    "C4": "제품명·미승인", "C5": "가격·할인", "C6": "대가성 후기·추천",
    "C7": "인증·자격 표기", "C8": "필수정보·리스크 누락", "C9": "기타",
}
FLAG = {"JP": "🇯🇵", "KR": "🇰🇷"}


def highlight(text, spans):
    """문구 안의 위반 span을 <mark>로 감싼다. 긴 span부터 처리해 겹침을 피한다."""
    text = html.escape(text)
    for span in sorted(set(spans), key=len, reverse=True):
        esc = html.escape(span)
        text = text.replace(esc, f"<mark>{esc}</mark>")
    return text


def cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def render(rows):
    out = ["# 평가셋 (golden) 보기용", "",
           "> `golden.jsonl`에서 자동 생성. 직접 고치지 말고 jsonl을 고친 뒤 `python3 eval/render.py`를 다시 실행한다.", ""]

    verdicts = collections.Counter(r["verdict"] for r in rows)
    out += ["| 판정 | 건수 |", "| --- | --- |"]
    out += [f"| {BADGE[v]} | {verdicts[v]} |" for v in ("violation", "caution", "ok")]
    out += ["", "## 목차", ""]
    out += [f"- [{r['id']}](#{r['id']}) {BADGE[r['verdict']]} {cell(r['title'])}" for r in rows]
    out.append("")

    for r in rows:
        spans_by_field = collections.defaultdict(list)
        for f in r["findings"]:
            if f.get("span"):
                spans_by_field[f["field"]].append(f["span"])

        out += ["---", "", f'<a id="{r["id"]}"></a>',
                f"## {r['id']} · {BADGE[r['verdict']]}", "",
                f"`{r['lang']}` · `{r['label_status']}` · `{r['origin']}`", "",
                f"**제목**　{highlight(r['title'], spans_by_field['title'])}", "",
                f"**설명**　{highlight(r['description'], spans_by_field['description'])}", ""]

        if r["findings"]:
            out += ["| # | 문구 | 유형 | 국가 | 등급 | 근거 |", "| --- | --- | --- | --- | --- | --- |"]
            for i, f in enumerate(r["findings"], 1):
                span = f"「{cell(f['span'])}」" if f.get("span") else "*(누락)*"
                out.append(f"| {i} | {span} | {f['category']} {CATEGORY.get(f['category'], '')} "
                           f"| {FLAG.get(f['country'], f['country'])} | {BADGE[f['severity']]} | {cell(f['rule'])} |")
            out.append("")
            for i, f in enumerate(r["findings"], 1):
                out += [f"**{i}.** {f['reason']}", "", f"> ✏️ 수정안: {f['suggestion']}", ""]
        else:
            out += ["위반·주의 문구 없음", ""]

        if r.get("note"):
            out += [f"📝 {r['note']}", ""]
    return "\n".join(out)


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    OUT.write_text(render(rows))
    print(f"{OUT.relative_to(ROOT)} 생성 ({len(rows)}건)")


if __name__ == "__main__":
    main()
