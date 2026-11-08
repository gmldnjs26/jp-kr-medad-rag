"""Phase 2: 원본 자료를 검색 단위(chunk)로 자른다. 대상 자료는 SOURCES (2026-11-07 재선정).

자료 종류마다 자르는 단위가 다르다.
- 법령 (JP 施行規則 XML, KR 법령 txt): 号/호 단위. 号가 없으면 項/항 단위.
  호만 떼면 무엇이 금지인지 사라지므로, 앞에 「법령명 조 제목 + 항의 머리말」을 붙인다.
- 医療広告等ガイドライン: 목차(第3-1 › 1 › (3)) 단위. 정답이 「第3-1 1(3)」 형식으로 인용한다.
- 그 밖의 가이드라인·Q&A·지침 PDF: 1차로 페이지 단위. 그 페이지 시점의 제목을 앞에 붙인다.
너무 긴 chunk는 MAX_CHARS 이하로 문장 경계에서 나눈다.

사용법: .venv/bin/python src/chunk.py   → data/chunks/chunks.jsonl
"""
import json
import pathlib
import re
import sys
import unicodedata

from pypdf import PdfReader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_corpus import RAW, jp_articles, kr_articles, pdf_text  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/chunks/chunks.jsonl"
MAX_CHARS = 1500

KANJI = dict(zip("〇一二三四五六七八九", range(10)))
KNUM = "〇一二三四五六七八九十百"


def kanji_to_int(s):
    total, num = 0, 0
    for ch in s:
        if ch in "百十":
            total += (num or 1) * (100 if ch == "百" else 10)
            num = 0
        else:
            num = KANJI[ch]
    return total + num


def arabic(s):
    """第六条の五 → 第6条の5"""
    return re.sub(f"[{KNUM}]+", lambda m: str(kanji_to_int(m.group())), s)


def split_long(text, limit=MAX_CHARS):
    """limit을 넘으면 문장 끝(。. 다.)에서 나눈다."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[。．])|(?<=다\.)\s*|\n", text):
        if cur and len(cur) + len(sent) > limit:
            parts.append(cur.strip())
            cur = ""
        cur += sent if sent.endswith(("。", "．")) else sent + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


# ---------------------------------------------------------------- 일본 법령

JP_NOISE = re.compile(r"^(\d+|第 ?\d+ ?回 .*|参考資料.*|令和 ?\d+ ?年 ?\d+ ?月 ?\d+ ?日)\s*$")
JP_ART = re.compile(rf"^第([{KNUM}]+条(?:の[{KNUM}]+)*)(（[^）]*）)?[ 　]?(.*)$")
JP_PARA = re.compile(r"^([２-９]|[1-9][0-9]?|[１-９][０-９]?)[ 　](.*)$")
JP_ITEM = re.compile(rf"^([{KNUM}]+)[ 　](.*)$")
JP_LAWS = [("医療法施行令", "医療法施行令"), ("医療法施行規則", "医療法施行規則"),
           ("告示第108号", "告示108号"), ("医療法", "医療法")]


def jp_law_units(lines, law):
    """일본 법령 텍스트를 [(법령명, 조, 조 제목, 항 번호, 항 머리말, 号 번호|None, 本文)]로 바꾼다.
    PDF에서 줄바꿈된 문장은 이어 붙인다. 附則은 건너뛴다.
    """
    units, art, caption, para, lead, item, buf = [], None, "", 1, "", None, ""
    skip = False

    def flush():
        nonlocal buf
        if art and buf.strip() and not skip:
            units.append((law, art, caption, para, lead, item, buf.strip()))
        buf = ""

    pending_caption = ""
    for raw in lines:
        line = raw.strip()
        if not line or JP_NOISE.match(line):
            continue
        if line[0] in "〇○":                                   # 법령 머리말 (告示는 여러 줄)
            flush()
            law = next((v for k, v in JP_LAWS if k in line.replace("１０８", "108")), law)
            art, skip = None, False
            continue
        if re.match(r"^附\s*則", line):
            flush()
            skip = True
            continue
        if re.match(r"^（[^）]+）$", line):                   # 다음 조의 제목
            pending_caption = line
            continue
        if m := JP_ART.match(line):
            flush()
            art, caption, para, item = m.group(1), m.group(2) or pending_caption, 1, None
            pending_caption, lead, buf = "", m.group(3), m.group(3)
            continue
        if art is None:
            continue                                           # 告示 머리말의 이어지는 줄
        if (m := JP_PARA.match(line)) and not m.group(2).startswith(("号", "条", "項", "年")):
            flush()
            para = int(unicodedata.normalize("NFKC", m.group(1)))
            item, lead, buf = None, m.group(2), m.group(2)
            continue
        if m := JP_ITEM.match(line):
            if item is None:                                   # 첫 号가 나오면 지금까지가 항의 머리말
                lead = buf.strip()
                buf = ""
            else:
                flush()
            item, buf = m.group(1), m.group(2)
            continue
        buf += line                                            # 줄바꿈된 문장 잇기
        if item is None:
            lead = buf
    flush()
    return units


def jp_law_chunks(text, law, source):
    out = []
    units = jp_law_units(text.splitlines(), law)
    has_items = {(u[0], u[1], u[3]) for u in units if u[5]}
    multi_para = {(u[0], u[1]) for u in units if u[3] > 1}     # 항이 하나뿐인 조는 「第1項」을 쓰지 않는다
    for law, art, caption, para, lead, item, body in units:
        if item is None and (law, art, para) in has_items:
            continue                                           # 号가 있는 항은 号 chunk에 머리말로 들어간다
        show_para = para > 1 or (item and (law, art) in multi_para)
        ref = arabic(f"{law} 第{art}") + (f"第{para}項" if show_para else "") + (f"第{kanji_to_int(item)}号" if item else "")
        head = f"[JP] {law} 第{art}{caption}" + (f" 第{para}項" if para > 1 else "")
        text = f"{head}\n{lead}\n{item} {body}" if item else f"{head}\n{body}"
        out.append(dict(country="JP", doc=law, ref=ref, source_file=source, text=text))
    return out


# ---------------------------------------------------------------- 한국 법령

KR_ART = re.compile(r"^제(\d+)조(?:의(\d+))?\(([^)]*)\)\s*(.*)$")
KR_PARA = re.compile(r"^([①-⑳])\s*(.*)$")
KR_ITEM = re.compile(r"^(\d+)\.\s*(.*)$")
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def kr_law_chunks(text, law, source):
    out = []
    art = title = lead = None
    para, item, buf = 0, None, ""

    def flush():
        nonlocal buf
        body = re.sub(r"<(개정|신설|전문개정)[^>]*>", "", buf).strip()
        if art and body:
            out.append((art, title, para, lead, item, body))
        buf = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line or re.match(r"^제\d+(절|장)\s", line) or "삭제 <" in line[:20]:
            continue
        if m := KR_ART.match(line):
            flush()
            art = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
            title, para, item, lead = m.group(3), 0, None, ""
            line = m.group(4)
            if not line:
                continue
        if m := KR_PARA.match(line):
            flush()
            para, item = CIRCLED.index(m.group(1)) + 1, None
            lead = buf = m.group(2)
            continue
        if (m := KR_ITEM.match(line)) and art:
            if item is None:
                lead = buf.strip()
                buf = ""
            else:
                flush()
            item, buf = m.group(1), m.group(2)
            continue
        buf += (" " if buf else "") + line
        if item is None:
            lead = buf
    flush()

    has_items = {(a, p) for a, _, p, _, i, _ in out if i}
    chunks = []
    for art, title, para, lead, item, body in out:
        if item is None and (art, para) in has_items:
            continue
        ref = f"{law} {art}" + (f"제{para}항" if para else "") + (f"제{item}호" if item else "")
        head = f"[KR] {law} {art}({title})" + (f" {CIRCLED[para - 1]}" if para else "")
        text = f"{head}\n{re.sub(r'<(개정|신설)[^>]*>', '', lead).strip()}\n{item}. {body}" if item else f"{head}\n{body}"
        chunks.append(dict(country="KR", doc=law, ref=ref, source_file=source, text=text))
    return chunks


# ---------------------------------------------------------------- 일본 가이드라인 (목차 단위)

GL_H1 = re.compile(r"^第([0-9]+(?:-[0-9]+)?)\s+(.+)$")
GL_H2 = re.compile(r"^([0-9]+)\s+(.+)$")
GL_H3 = re.compile(r"^\(([0-9]+)\)\s*(.+)$")


def is_heading(nfkc_line):
    return len(nfkc_line) <= 45 and not nfkc_line.endswith(("。", "、")) and "...." not in nfkc_line


def jp_guideline_chunks(text, doc, source):
    out = []
    h = {1: None, 2: None, 3: None}       # (번호, 제목)
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body or h[1] is None:
            return
        ref = f"{doc} 第{h[1][0]}" + (f" {h[2][0]}" if h[2] else "") + (f"({h[3][0]})" if h[3] else "")
        path = " > ".join(f"{fmt.format(x[0])} {x[1]}" for fmt, x in (("{}", h[1]), ("{}", h[2]), ("({})", h[3])) if x)
        for i, part in enumerate(split_long(body)):
            out.append(dict(country="JP", doc=doc, ref=ref, source_file=source,
                            text=f"[JP] {doc} 第{path}" + (f" (続き{i})" if i else "") + f"\n{part}"))

    for raw in text.splitlines():
        line = raw.strip()
        n = unicodedata.normalize("NFKC", line).replace("－", "-")
        if re.match(r"^-\s*\d+\s*-$", n):                    # ページ番号
            continue
        # ①②…는 NFKC에서 1, 2…가 되어 GL_H2 제목처럼 보인다. 원문이 원 숫자로 시작하면 본문 목록이다
        # (2026-11-07 수정 전에는 第3-3 2 限定解除 4요건 등 목록 본문이 제목으로 사라졌다)
        # 第3-1의 「(7) 治療等の内容又は効果について…写真等」는 47자라 45자 기준을 넘는다. 이 절에서만 50자까지 제목으로 본다.
        # (다른 절, 특히 第3-2의 広告可能事項 목록에는 45~50자의 본문 줄이 있어 기준을 전체로 늘리지 않는다)
        long_h3 = h[1] and h[1][0] == "3-1" and GL_H3.match(n) and len(n) <= 50 and not n.endswith(("。", "、"))
        if (is_heading(n) or long_h3) and not re.match(r"^[①-⑳]", line):
            for level, pat in ((1, GL_H1), (3, GL_H3), (2, GL_H2)):
                if m := pat.match(n):
                    flush()
                    h[level] = (m.group(1), m.group(2).strip())
                    for deeper in range(level + 1, 4):
                        h[deeper] = None
                    break
            else:
                buf.append(line)
            continue
        buf.append(line)
    flush()
    return out


# ---------------------------------------------------------------- 페이지 단위 (事例解説書, 복지부 가이드라인)

def page_texts(rel):
    for i, page in enumerate(PdfReader(RAW / rel).pages, 1):
        t = re.sub(r"[ \t　]+", " ", page.extract_text() or "")
        yield i, re.sub(r"\n\s*\n+", "\n", t).strip()


# 제목 후보: 로마숫자(Ⅰ.), 「第1」, 「Q1」/「Q 1-1」, 「1.」 「1)」로 시작하는 짧은 줄
PAGE_HEADING = re.compile(r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\.?|第\s*\d+|Q\s*\d+(?:-\d+)?|\d+[.)])\s*\S")


def page_chunks(rel, doc, country, source):
    """페이지 단위. 헤더에 그 페이지까지 마지막으로 본 제목을 붙인다.
    구조 단위 파서를 만들기 전의 1차 청킹이다 (2026-11-07). 검색 성적이 나쁜 문서부터 구조 단위로 바꾼다."""
    out, heading = [], ""
    for page, text in page_texts(rel):
        if len(text) < 30:  # 표지·이미지 페이지
            continue
        for line in text.splitlines():
            if PAGE_HEADING.match(line) and len(line.strip()) <= 50 and "...." not in line and "···" not in line:
                heading = line.strip()
        for part in split_long(text):
            out.append(dict(country=country, doc=doc, ref=f"{doc} p.{page}", page=page, source_file=source,
                            text=f"[{country}] {doc} p.{page} {heading}".rstrip() + f"\n{part}"))
    return out


# ---------------------------------------------------------------- 목차 기반 절 단위 (식약처 2종, 2026-11-08)

def _flat(s):
    """제목 비교용: 공백과 가운뎃점·따옴표를 지운다 (PDF마다 ㆍ‧· 표기가 다르다)"""
    return re.sub(r"[\s·ㆍ‧・“”\"']", "", s)


def section_chunk(doc, country, source, path, pages, body):
    out = []
    span = f"p.{pages[0]}" + (f"-{pages[-1]}" if len(pages) > 1 else "")
    for i, part in enumerate(split_long(body)):
        out.append(dict(country=country, doc=doc, ref=f"{doc} {path.split(' > ')[-1]}", page=pages[0], pages=pages,
                        source_file=source,
                        text=f"[{country}] {doc} > {path} ({span})" + (f" (続き{i})" if i else "") + f"\n{part}"))
    return out


def toc_title_chunks(rel, doc, country, source, toc, body_from):
    """목차 제목 목록(toc: [(상위 제목, 번호, 제목)])을 본문에서 순서대로 찾아 그 줄에서 자른다.
    순서대로만 찾으므로 본문 안의 「1. 신문·방송 또는 잡지」 같은 법조문 목록을 제목으로 오인하지 않는다."""
    out, k, cur, buf, pages = [], 0, None, [], []

    def flush():
        if cur and buf:
            out.extend(section_chunk(doc, country, source, cur, sorted(set(pages)), "\n".join(buf).strip()))

    for page, text in page_texts(rel):
        if page < body_from:
            continue
        for line in text.splitlines():
            if k < len(toc):
                part, num, title = toc[k]
                if _flat(line).startswith(_flat(f"{num}.{title}")[:8]):
                    flush()
                    cur, buf, pages, k = f"{part} > {num}. {title}", [], [], k + 1
            if cur:
                buf.append(line)
                pages.append(page)
    flush()
    if k < len(toc):
        raise SystemExit(f"{rel}: 목차 제목을 본문에서 못 찾음: {toc[k]}")
    return out


def toc_page_chunks(rel, doc, country, source, toc_pages):
    """목차에 「번호. 제목 쪽」이 있는 문서. 인쇄 쪽 = pypdf 순번이라 시작 쪽부터 다음 절 직전 쪽까지를 한 절로 묶는다.
    목차에서 절 묶음 다음에 나오는 줄이 그 묶음의 장 제목이다 (예: 「의료기기법 위반 광고 식별 요령」 다음 「Chapter 1」)."""
    texts = dict(page_texts(rel))
    entries, group = [], []
    for p in toc_pages:
        for line in texts[p].splitlines():
            if m := re.match(r"^\s*(\d{1,2})\.\s*(.+?)\s+(\d{1,3})\s*$", line):
                group.append((int(m.group(1)), m.group(2).strip(), int(m.group(3))))
            elif group and line.strip() and not line.strip().startswith(("Chapter", "Cha", "Contents")):
                entries += [(line.strip(), n, t, s) for n, t, s in group]
                group = []
    entries += [("", n, t, s) for n, t, s in group]
    last = max(texts)
    out = []
    for i, (chap, n, title, start) in enumerate(entries):
        end = (entries[i + 1][3] - 1) if i + 1 < len(entries) else last
        pages = [q for q in range(start, max(start, end) + 1) if len(texts.get(q, "")) >= 30]
        body = "\n".join(texts[q] for q in pages)
        if body:
            out.extend(section_chunk(doc, country, source, f"{chap} > {n}. {title}".lstrip(" >"), pages, body))
    return out


MFDS_DRUG_TOC = [("I. 일반사항", 1, "목적"), ("I. 일반사항", 2, "관련 법령"), ("I. 일반사항", 3, "적용 범위"),
                 ("I. 일반사항", 4, "의약품 광고란")] + [("II. 의약품광고 가이드라인", i + 1, t) for i, t in enumerate([
                     "거짓ㆍ과장광고", "의ㆍ약전문가 추천광고", "근거문헌 인용 광고", "소비자 오인 우려가 있는 광고",
                     "현상품ㆍ사은품 등 경품류 제공광고", "비방광고", "체험담을 이용하는 광고", "저속하거나 혐오감을 주는 광고",
                     "의약품 오남용 우려 광고", "보증 및 최고·최상 절대적 표현 광고", "안전성을 강조하는 광고",
                     "특정대상자를 한정하는 광고", "의약품을 의약품이 아닌 것으로 오인우려 광고", "주성분이 아닌 성분 광고",
                     "인터넷 및 그와 유사한 매체를 통한 광고", "방송법에 따른 간접광고", "기사형 광고 및 보도자료",
                     "소비자 대상 강연 등", "사전피임제 대중광고"])] + [("III. 전문의약품 정보제공 가이드라인", i + 1, t) for i, t in enumerate([
                         "전문의약품 광고 제한적 허용", "전문가 대상 전문의약품 정보제공 일반사항",
                         "인터넷 “자사 홈페이지”에서 정보제공", "복약지도에 도움이 되는 환자용 사용설명서"])]


# ---------------------------------------------------------------- 실행

def jp_xml_text(rel, numbers):
    return jp_articles(rel, numbers)


SOURCES = [  # 2026-11-07 자료 재선정 (docs/phase2-retrieval-log.md 7절)
    # 일본
    ("jp/医療法施行規則_323M40000100050_施行20261001.xml",
     lambda p: jp_law_chunks(jp_xml_text(p, ["1_9", "1_9_2"]), "医療法施行規則", p)),
    ("jp/医療広告等ガイドライン_最終改正20260330.pdf",
     lambda p: jp_guideline_chunks(pdf_text(p), "医療広告等ガイドライン", p)),
    ("jp/医療広告等ガイドラインQA_最終改正20260330.pdf",
     lambda p: page_chunks(p, "医療広告ガイドラインQ&A", "JP", p)),
    ("jp/景表法_不当な価格表示についての考え方.pdf",
     lambda p: page_chunks(p, "不当な価格表示についての考え方", "JP", p)),
    ("jp/景表法_ステマ告示運用基準_20230328.pdf",
     lambda p: page_chunks(p, "ステマ告示運用基準", "JP", p)),
    # 한국
    ("kr/의료법_20261002.txt", lambda p: kr_law_chunks(kr_articles(p, ["27조", "56조"]), "의료법", p)),
    ("kr/의료법시행령_20260210.txt", lambda p: kr_law_chunks(kr_articles(p, ["23조"]), "의료법 시행령", p)),
    ("kr/의료해외진출법_20260512.txt", lambda p: kr_law_chunks(kr_articles(p, ["6조", "15조"]), "의료해외진출법", p)),
    ("kr/KHIDI_외국인환자유치온라인의료광고가이드라인_20210312.pdf",
     lambda p: page_chunks(p, "KHIDI 외국인환자유치 온라인 의료광고 가이드라인", "KR", p)),
    ("kr/의협_의료광고심의위원회_사전자율심의기준_191226_v3.pdf",
     lambda p: page_chunks(p, "의협 의료광고 사전자율심의기준", "KR", p)),
    ("kr/식약처_의약품광고및전문의약품정보제공가이드라인_20221130.pdf",
     lambda p: toc_title_chunks(p, "식약처 의약품 광고 가이드라인", "KR", p, MFDS_DRUG_TOC, body_from=6)),
    ("kr/식약처_의료기기법위반광고해설_20170523.pdf",
     lambda p: toc_page_chunks(p, "식약처 의료기기법 위반광고 해설", "KR", p, toc_pages=[4, 5])),
    ("kr/복지부_건강한의료광고_유형별사례및체크리스트_2판_20251222.pdf",
     lambda p: page_chunks(p, "복지부 유형별 의료광고 사례 및 체크리스트", "KR", p)),
]


def main():
    chunks = []
    for rel, fn in SOURCES:
        got = fn(rel)
        for c in got:
            c["source_file"] = f"data/raw/{rel}"
        sizes = [len(c["text"]) for c in got]
        print(f"{len(got):>5} chunks  평균 {sum(sizes) // max(len(sizes), 1):>5}자  최대 {max(sizes, default=0):>5}자  {rel}")
        chunks += got
    for i, c in enumerate(chunks):
        c["id"] = f"c{i:05d}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks))
    print(f"합계 {len(chunks)} chunks, {sum(len(c['text']) for c in chunks):,}자 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
