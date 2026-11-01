"""예측 결과(predictions.jsonl)를 정답(golden.jsonl)과 비교해 점수를 낸다.

사용법: .venv/bin/python eval/score.py eval/results/<run>/predictions.jsonl
같은 폴더에 summary.md를 쓴다.

지표
- verdict 정확도: 3단계 판정이 정답과 같은 비율
- 위반 탐지 재현율: 정답이 violation/caution인 건을 ok가 아니라고 잡은 비율 (놓치면 안 되는 지표)
- 오탐률: 정답이 ok인 건을 violation/caution으로 잡은 비율
- 문구(span) 재현율: 정답 finding의 span을 같은 나라로 찾았는지 (부분 일치 허용)
- 문구(span) 적합률: 예측 finding의 span이 같은 나라의 정답 span과 맞는 비율 (정답에 없는 지적은 감점)
- 근거 조문 일치율: 찾은 finding 중 법령명과 조문 번호가 하나라도 겹치는 비율
- C8(누락) 재현율·적합률: span이 없는 C8을 (샘플, 나라) 단위로 비교
"""
import collections
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEVEL = {"ok": 0, "caution": 1, "violation": 2}
KANJI = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
SEP = r"[·・,、]\s*"
JP_ART = re.compile(r"第(\d+)条((?:の\d+)*)(?:第(\d+)項)?(?:第(\d+)号((?:" + SEP + r"第\d+号)*))?")
KR_ART = re.compile(r"제(\d+)조(?:의(\d+))?(?:제(\d+)항)?(?:제(\d+)호((?:" + SEP + r"제\d+호)*))?")
GL_ITEM = re.compile(r"第3-(\d+)(?:\s*(\d+))?(?:\s*\((\d+)\))?((?:" + SEP + r"\d+\s*\(\d+\))*)")
# 법령명 표기 → 정규 이름. 긴 이름부터 찾는다 (의료법 시행령이 의료법으로 잡히지 않게).
LAWS = [
    (r"医療法施行規則|施行規則", "医療法施行規則"),
    (r"医療法施行令|施行令", "医療法施行令"),
    (r"医薬品[、,]?\s*医療機器等の品質[^第]*?法律|医薬品医療機器等法|薬機法", "薬機法"),
    (r"不当景品類及び不当表示防止法|景品表示法|景表法", "景表法"),
    (r"医療法", "医療法"),
    (r"의료법\s*시행령|시행령", "의료법시행령"),
    (r"의료법\s*시행규칙|시행규칙", "의료법시행규칙"),
    (r"의료\s*해외진출\s*및\s*외국인환자\s*유치\s*지원에\s*관한\s*법률|의료해외진출법", "의료해외진출법"),
    (r"표시[·・]?\s*광고의\s*공정화에\s*관한\s*법률|표시광고법", "표시광고법"),
    (r"약사법", "약사법"),
    (r"의료법", "의료법"),
]
LAW_RE = re.compile("|".join(f"(?P<l{i}>{pat})" for i, (pat, _) in enumerate(LAWS)))


def kanji_to_int(s):
    """六十八 → 68, 百二十 → 120. 한자 숫자 한 덩어리를 정수로 바꾼다."""
    total, num = 0, 0
    for ch in s:
        if ch == "百":
            total += (num or 1) * 100; num = 0
        elif ch == "十":
            total += (num or 1) * 10; num = 0
        else:
            num = KANJI[ch]
    return total + num


def normalize(rule):
    """전각 숫자·원문자(②)를 반각으로, 한자 숫자를 아라비아 숫자로 바꾼다."""
    s = unicodedata.normalize("NFKC", rule or "")
    return re.sub(r"[〇一二三四五六七八九十百]+", lambda m: str(kanji_to_int(m.group())), s)


def law_at(s, pos):
    """pos보다 앞에서 가장 가까운 법령명. 없으면 "?" (어느 법령과도 맞는 것으로 본다)."""
    law = "?"
    for m in LAW_RE.finditer(s):
        if m.start() >= pos:
            break
        law = LAWS[int(m.lastgroup[1:])][1]
    return law


def articles(rule):
    """rule 문자열에서 「법령명:조문 키」를 뽑는다. 항·호까지 살린다.
    医療法 第六条の五第二項第一号 → 医療法:条6の5-2-1 / 의료법 제56조제2항제14호 → 의료법:조56-2-14
    医療広告等ガイドライン 第3-1 1(5) → GL:GL3-1-1-5
    """
    s = normalize(rule)
    keys = set()

    def add(law, head, parts, listed):
        """listed: 「·제8호」「・1(8)」처럼 같은 상위 번호 아래 나열된 하위 번호"""
        parts = [x for x in parts if x]
        keys.add(f"{law}:" + "-".join([head] + parts))
        for extra in re.findall(r"\d+(?:\s*\(\d+\))?", listed or ""):
            nums = re.findall(r"\d+", extra)
            tail = parts[:-len(nums)] + nums if len(parts) >= len(nums) else nums
            keys.add(f"{law}:" + "-".join([head] + tail))

    for m in JP_ART.finditer(s):
        add(law_at(s, m.start()), "条" + m.group(1) + m.group(2), m.group(3, 4), m.group(5))
    for m in KR_ART.finditer(s):
        add(law_at(s, m.start()), "조" + m.group(1) + (f"의{m.group(2)}" if m.group(2) else ""), m.group(3, 4), m.group(5))
    for m in GL_ITEM.finditer(s):
        add("GL", "GL3", (m.group(1),) + m.group(2, 3), m.group(4))
    return keys


def article_match(gold_rule, pred_rule):
    """법령명이 같고 조문 키가 하나라도 맞으면 일치. 한쪽이 더 세밀하면 상위 번호가 같을 때 일치로 본다.
    예: 정답 条68 / 예측 条68-1 → 일치, 정답 조56-2-14 / 예측 조56-2-2 → 불일치,
        정답 景表法 第5条 / 예측 医療法 第5条 → 불일치. 법령명을 못 찾은 쪽("?")은 법령명 비교를 건너뛴다.
    """
    for g in articles(gold_rule):
        gl, gk = g.split(":", 1)
        for p in articles(pred_rule):
            pl, pk = p.split(":", 1)
            if gl != pl and "?" not in (gl, pl):
                continue
            if gk == pk or gk.startswith(pk + "-") or pk.startswith(gk + "-"):
                return True
    return False


def span_match(gold, pred):
    """한쪽이 다른 쪽을 포함하거나, 가장 긴 공통 부분 문자열이 짧은 쪽의 절반 이상이면 일치로 본다.
    예: 정답 「通訳スタッフがお肌の状態を診断し」 / 예측 「日本語通訳スタッフがお肌の状態を診断」 → 일치
    """
    if not (gold and pred):
        return False
    if gold in pred or pred in gold:
        return True
    best = 0
    prev = [0] * (len(pred) + 1)
    for a in gold:
        cur = [0] * (len(pred) + 1)
        for j, b in enumerate(pred, 1):
            if a == b:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best >= min(len(gold), len(pred)) / 2


def score(golden, preds):
    rows = []
    for g in golden:
        p = preds.get(g["id"])
        if p is None:
            continue
        rows.append((g, p))
    n = len(rows)
    verdict_ok = sum(g["verdict"] == p["verdict"] for g, p in rows)
    confusion = collections.Counter((g["verdict"], p["verdict"]) for g, p in rows)

    flagged_gold = [(g, p) for g, p in rows if g["verdict"] != "ok"]
    ok_gold = [(g, p) for g, p in rows if g["verdict"] == "ok"]
    detect = sum(p["verdict"] != "ok" for _, p in flagged_gold)
    false_pos = sum(p["verdict"] != "ok" for _, p in ok_gold)

    span_total = span_hit = rule_hit = 0
    per_cat = collections.defaultdict(lambda: [0, 0])
    misses = []
    for g, p in rows:
        for gf in g["findings"]:
            if gf["span"] is None:
                continue
            span_total += 1
            per_cat[gf["category"]][1] += 1
            hit = next((pf for pf in p["findings"]
                        if pf["country"] == gf["country"] and span_match(gf["span"], pf.get("span"))), None)
            if hit:
                span_hit += 1
                per_cat[gf["category"]][0] += 1
                if article_match(gf["rule"], hit["rule"]):
                    rule_hit += 1
            else:
                misses.append((g["id"], gf["country"], gf["category"], gf["span"]))

    # 적합률: 예측 span이 같은 나라의 정답 span 중 하나와 맞는가. 정답에 없는 지적은 extras로 모은다.
    pred_total = pred_hit = 0
    extras = []
    for g, p in rows:
        gold_spans = [(gf["country"], gf["span"]) for gf in g["findings"] if gf["span"]]
        for pf in p["findings"]:
            if not pf.get("span"):
                continue
            pred_total += 1
            if any(c == pf["country"] and span_match(sp, pf["span"]) for c, sp in gold_spans):
                pred_hit += 1
            else:
                extras.append((g["id"], pf["country"], pf["category"], pf["span"]))

    # C8(누락): span 없는 C8을 (샘플, 나라) 단위로 비교
    gold_c8 = {(g["id"], gf["country"]) for g, _ in rows for gf in g["findings"]
               if gf["category"] == "C8" and not gf["span"]}
    pred_c8 = {(g["id"], pf["country"]) for g, p in rows for pf in p["findings"]
               if pf["category"] == "C8" and not pf.get("span")}

    return dict(n=n, verdict_ok=verdict_ok, confusion=confusion, detect=detect, flagged=len(flagged_gold),
                false_pos=false_pos, ok_total=len(ok_gold), span_total=span_total, span_hit=span_hit,
                rule_hit=rule_hit, per_cat=per_cat, misses=misses,
                pred_total=pred_total, pred_hit=pred_hit, extras=extras,
                c8_gold=len(gold_c8), c8_pred=len(pred_c8), c8_hit=len(gold_c8 & pred_c8),
                c8_missed=sorted(gold_c8 - pred_c8), c8_extra=sorted(pred_c8 - gold_c8),
                wrong=[(g["id"], g["verdict"], p["verdict"]) for g, p in rows if g["verdict"] != p["verdict"]])


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def report(s, meta):
    lines = [f"# Phase 1 결과: {meta.get('run', '')}", "",
             f"- 모델: `{meta.get('model')}` / 샘플: {s['n']}건",
             f"- 토큰: 입력 {meta.get('prompt', 0):,} (그중 캐시 {meta.get('cached', 0):,}) / 출력 {meta.get('output', 0):,}",
             "", "| 지표 | 값 |", "| --- | --- |",
             f"| verdict 정확도 | {pct(s['verdict_ok'], s['n'])} |",
             f"| 위반 탐지 재현율 (정답 violation·caution → ok 아님) | {pct(s['detect'], s['flagged'])} |",
             f"| 오탐률 (정답 ok → violation·caution) | {pct(s['false_pos'], s['ok_total'])} |",
             f"| 문구(span) 재현율 | {pct(s['span_hit'], s['span_total'])} |",
             f"| 문구(span) 적합률 (예측 중 정답과 맞는 것) | {pct(s['pred_hit'], s['pred_total'])} |",
             f"| 근거 조문 일치율 (찾은 문구 중, 법령명 포함) | {pct(s['rule_hit'], s['span_hit'])} |",
             f"| C8(누락) 재현율 (샘플×나라) | {pct(s['c8_hit'], s['c8_gold'])} |",
             f"| C8(누락) 적합률 (샘플×나라) | {pct(s['c8_hit'], s['c8_pred'])} |",
             "", "## 혼동 행렬 (행: 정답, 열: 예측)", "",
             "| 정답 \\ 예측 | ok | caution | violation |", "| --- | --- | --- | --- |"]
    for gv in LEVEL:
        lines.append(f"| {gv} | " + " | ".join(str(s["confusion"][(gv, pv)]) for pv in LEVEL) + " |")
    lines += ["", "## 카테고리별 문구 재현율", "", "| 카테고리 | 재현율 |", "| --- | --- |"]
    for cat in sorted(s["per_cat"], key=lambda c: int(c[1:])):
        hit, tot = s["per_cat"][cat]
        lines.append(f"| {cat} | {pct(hit, tot)} |")
    lines += ["", "## verdict가 틀린 건", ""]
    lines += [f"- {i}: 정답 {gv} → 예측 {pv}" for i, gv, pv in s["wrong"]] or ["- 없음"]
    lines += ["", "## 놓친 문구", ""]
    lines += [f"- {i} [{c}] {cat} 「{sp}」" for i, c, cat, sp in s["misses"]] or ["- 없음"]
    lines += ["", "## 정답에 없는 지적 (모델 과잉 또는 정답 누락, 하나씩 확인)", ""]
    lines += [f"- {i} [{c}] {cat} 「{sp}」" for i, c, cat, sp in s["extras"]] or ["- 없음"]
    lines += ["", "## C8(누락) 차이", ""]
    lines += [f"- 놓침: {i} [{c}]" for i, c in s["c8_missed"]]
    lines += [f"- 정답에 없음: {i} [{c}]" for i, c in s["c8_extra"]]
    if not (s["c8_missed"] or s["c8_extra"]):
        lines.append("- 없음")
    return "\n".join(lines) + "\n"


def main():
    pred_path = pathlib.Path(sys.argv[1])
    golden = [json.loads(l) for l in (ROOT / "eval/golden.jsonl").read_text().splitlines() if l.strip()]
    preds, meta = {}, {}
    for line in pred_path.read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        preds[r["id"]] = r["prediction"]
        for k in ("prompt", "cached", "output"):
            meta[k] = meta.get(k, 0) + r["usage"][k]
        meta["model"] = r["model"]
    meta["run"] = pred_path.parent.name
    md = report(score(golden, preds), meta)
    (pred_path.parent / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
