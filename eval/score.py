"""예측 결과(predictions.jsonl)를 정답(golden.jsonl)과 비교해 점수를 낸다.

사용법: .venv/bin/python eval/score.py eval/results/<run>/predictions.jsonl
같은 폴더에 summary.md를 쓴다.

지표
- verdict 정확도: 3단계 판정이 정답과 같은 비율
- 위반 탐지 재현율: 정답이 violation/caution인 건을 ok가 아니라고 잡은 비율 (놓치면 안 되는 지표)
- 오탐률: 정답이 ok인 건을 violation/caution으로 잡은 비율
- 문구(span) 재현율: 정답 finding의 span을 같은 나라로 찾았는지 (부분 일치 허용)
- 근거 조문 일치율: 찾은 finding 중 조문 번호가 하나라도 겹치는 비율
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


def articles(rule):
    """rule 문자열에서 조문 키를 뽑는다. 항·호까지 살린다.
    医療法 第六条の五第二項第一号 → 条6の5-2-1 / 의료법 제56조제2항제14호 → 조56-2-14
    医療広告等ガイドライン 第3-1 1(5) → GL3-1-1-5
    """
    s = normalize(rule)
    keys = set()

    def add(head, parts, listed):
        """listed: 「·제8호」「・1(8)」처럼 같은 상위 번호 아래 나열된 하위 번호"""
        parts = [x for x in parts if x]
        keys.add("-".join([head] + parts))
        for extra in re.findall(r"\d+(?:\s*\(\d+\))?", listed or ""):
            nums = re.findall(r"\d+", extra)
            keys.add("-".join([head] + parts[:-len(nums)] + nums) if len(parts) >= len(nums) else "-".join([head] + nums))

    for m in JP_ART.finditer(s):
        add("条" + m.group(1) + m.group(2), m.group(3, 4), m.group(5))
    for m in KR_ART.finditer(s):
        add("조" + m.group(1) + (f"의{m.group(2)}" if m.group(2) else ""), m.group(3, 4), m.group(5))
    for m in GL_ITEM.finditer(s):
        add("GL3", (m.group(1),) + m.group(2, 3), m.group(4))
    return keys


def article_match(gold_rule, pred_rule):
    """조문 키가 하나라도 맞으면 일치. 한쪽이 더 세밀하면 상위 번호가 같을 때 일치로 본다.
    예: 정답 条68 / 예측 条68-1 → 일치, 정답 조56-2-14 / 예측 조56-2-2 → 불일치
    한계: 법령명은 보지 않는다 (의료법 제27조와 약사법 제27조를 구분하지 못함).
    """
    for g in articles(gold_rule):
        for p in articles(pred_rule):
            if g == p or g.startswith(p + "-") or p.startswith(g + "-"):
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

    return dict(n=n, verdict_ok=verdict_ok, confusion=confusion, detect=detect, flagged=len(flagged_gold),
                false_pos=false_pos, ok_total=len(ok_gold), span_total=span_total, span_hit=span_hit,
                rule_hit=rule_hit, per_cat=per_cat, misses=misses,
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
             f"| 근거 조문 일치율 (찾은 문구 중) | {pct(s['rule_hit'], s['span_hit'])} |",
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
