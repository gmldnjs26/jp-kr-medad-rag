"""golden.jsonl 형식 검증과 분포 통계.

사용법: python3 eval/validate.py [eval/golden.jsonl]   (프로젝트 루트에서 실행)
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERDICTS = ["ok", "caution", "violation"]
CATEGORIES = {f"C{i}" for i in range(1, 10)}
REQUIRED = ["id", "split", "lang", "title", "description", "verdict", "findings", "label_status", "origin"]


def check(row, errors):
    rid = row.get("id", "?")
    for key in REQUIRED:
        if key not in row:
            errors.append(f"{rid}: '{key}' 없음")
    if row.get("verdict") not in VERDICTS:
        errors.append(f"{rid}: verdict 값 이상 {row.get('verdict')!r}")
    worst = "ok"
    for i, fd in enumerate(row.get("findings", [])):
        tag = f"{rid}.findings[{i}]"
        if fd.get("category") not in CATEGORIES:
            errors.append(f"{tag}: category 이상 {fd.get('category')!r}")
        if fd.get("country") not in ("JP", "KR"):
            errors.append(f"{tag}: country 이상 {fd.get('country')!r}")
        sev = fd.get("severity")
        if sev not in ("caution", "violation"):
            errors.append(f"{tag}: severity 이상 {sev!r}")
        elif VERDICTS.index(sev) > VERDICTS.index(worst):
            worst = sev
        span, field = fd.get("span"), fd.get("field")
        if span is None:
            if fd.get("category") != "C8":
                errors.append(f"{tag}: span null은 C8(누락)만 허용")
        elif field not in ("title", "description") or span not in row.get(field, ""):
            errors.append(f"{tag}: span {span!r}이 {field}에 글자 그대로 없음")
        src = fd.get("source_file")
        if not src or not (ROOT / src).exists():
            errors.append(f"{tag}: source_file 없음 {src!r}")
        for key in ("rule", "reason", "suggestion"):
            if not fd.get(key):
                errors.append(f"{tag}: '{key}' 비어 있음")
    if row.get("verdict") in VERDICTS and worst != row["verdict"]:
        errors.append(f"{rid}: verdict={row['verdict']} 인데 finding 최고 등급은 {worst}")


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "eval/golden.jsonl")
    rows, errors, ids = [], [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: JSON 오류 {e}")
            continue
        if row.get("id") in ids:
            errors.append(f"line {n}: id 중복 {row.get('id')}")
        ids.add(row.get("id"))
        check(row, errors)
        rows.append(row)

    total = len(rows)
    verdicts = collections.Counter(r.get("verdict") for r in rows)
    langs = collections.Counter(r.get("lang") for r in rows)
    cats = collections.Counter()
    for r in rows:
        cats.update({fd.get("category") for fd in r.get("findings", [])})
    print(f"샘플 {total}건")
    print("verdict:", ", ".join(f"{v} {verdicts[v]} ({verdicts[v] / total:.0%})" for v in VERDICTS))
    print("lang:", dict(langs))
    print("카테고리별 샘플 수:", ", ".join(f"{c} {cats[c]}" for c in sorted(CATEGORIES, key=lambda c: int(c[1:]))))
    if errors:
        print(f"\n오류 {len(errors)}건")
        print("\n".join(f"- {e}" for e in errors))
        sys.exit(1)
    print("\n오류 없음")


if __name__ == "__main__":
    main()
