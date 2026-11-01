"""Phase 1용 핵심 자료 묶음(data/corpus/core.md)을 만든다.

원본(data/raw)에서 판정에 필요한 법령 조문과 가이드라인 본문만 뽑아 하나의 텍스트로 합친다.
각 섹션 머리에 source_file 경로를 적어 두어, 모델이 근거 파일을 인용할 수 있게 한다.

사용법: .venv/bin/python src/build_corpus.py
"""
import pathlib
import re
import xml.etree.ElementTree as ET

from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/corpus/core.md"


def pdf_text(rel):
    reader = PdfReader(RAW / rel)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"[ \t　]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def kr_articles(rel, numbers):
    """law.go.kr 텍스트에서 '제N조(...)'로 시작하는 조문 블록을 뽑는다."""
    text = (RAW / rel).read_text()
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^\s*제(\d+조(?:의\d+)?)\(", text, re.M)]
    blocks = []
    for i, (pos, num) in enumerate(starts):
        if num in numbers:
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            blocks.append(re.sub(r"\n\s*\n+", "\n", text[pos:end]).strip())
    missing = set(numbers) - {n for _, n in starts}
    if missing:
        raise SystemExit(f"{rel}: 조문을 찾지 못함 {sorted(missing)}")
    return "\n\n".join(blocks)


def jp_articles(rel, numbers):
    """e-Gov 法令API XML의 본칙(MainProvision)에서 지정한 조문을 뽑는다."""
    root = ET.parse(RAW / rel).getroot()
    main = root.find(".//MainProvision")
    blocks = []
    for num in numbers:
        art = main.find(f".//Article[@Num='{num}']")
        if art is None:
            raise SystemExit(f"{rel}: Article {num} 없음")
        caption = "".join(art.findtext("ArticleCaption") or "")
        title = art.findtext("ArticleTitle") or ""
        lines = []
        for para in art.iter("Paragraph"):  # 項 번호(②)와 号 번호(二)를 살려야 모델이 조문을 정확히 인용한다
            num = para.findtext("ParagraphNum") or ""
            lines.append(f"{num} " + "".join("".join(s.itertext()) for s in para.find("ParagraphSentence").iter("Sentence")))
            for item in para.findall("Item"):
                lines.append(f"  {item.findtext('ItemTitle') or ''} " + "".join("".join(s.itertext()) for s in item.find("ItemSentence").iter("Sentence")))
        body = "\n".join(l.strip() for l in lines)
        blocks.append(f"{title}{caption}\n{body}")
    return "\n\n".join(blocks)


# (제목, source_file, 본문을 만드는 함수)
SECTIONS = [
    ("[JP] 医療法・施行令・施行規則・告示108号 広告関係規程抜粋",
     "jp/医療広告関係規程抜粋_医療法・施行令・施行規則・告示108号_mhlw.pdf",
     lambda p: pdf_text(p)),
    ("[JP] 医薬品医療機器等法 第66条〜第68条",
     "jp/薬機法_335AC0000000145_施行20260521.xml",
     lambda p: jp_articles(p, ["66", "67", "68"])),
    ("[JP] 医療広告等ガイドライン（最終改正 令和8年3月30日）",
     "jp/医療広告等ガイドライン_最終改正20260330.pdf",
     lambda p: pdf_text(p)),
    ("[JP] 医療広告規制におけるウェブサイト等の事例解説書 第6版",
     "jp/医療広告規制におけるウェブサイト等の事例解説書_第6版_20260330.pdf",
     lambda p: pdf_text(p)),
    ("[KR] 의료법 (제27조, 제56조, 제57조)",
     "kr/의료법_20261002.txt",
     lambda p: kr_articles(p, ["27조", "56조", "57조"])),
    ("[KR] 의료법 시행령 (제23조, 제24조)",
     "kr/의료법시행령_20260210.txt",
     lambda p: kr_articles(p, ["23조", "24조"])),
    ("[KR] 의료 해외진출 및 외국인환자 유치 지원에 관한 법률 (제6조, 제15조)",
     "kr/의료해외진출법_20260512.txt",
     lambda p: kr_articles(p, ["6조", "15조"])),
    ("[KR] 약사법 (제68조)",
     "kr/약사법_20261008.txt",
     lambda p: kr_articles(p, ["68조"])),
    ("[KR] 보건복지부 의료광고 가이드라인 (2020-07)",
     "kr/복지부_의료광고가이드라인_20200706.pdf",
     lambda p: pdf_text(p)),
]


def main():
    parts = ["# 의료광고 판정 자료 (Phase 1 core corpus)\n"]
    for title, rel, fn in SECTIONS:
        body = fn(rel)
        parts.append(f"\n\n========================================\n## {title}\n"
                     f"source_file: data/raw/{rel}\n========================================\n\n{body}")
        print(f"{len(body):>8,} chars  {title}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts))
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
