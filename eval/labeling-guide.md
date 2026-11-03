# 라벨링 가이드 (v0.2)

작성: 2026-10-31 / 대상: `eval/golden.jsonl`

평가셋 정답을 만드는 절차다. 이 정답으로 Phase 1(long-context baseline)과 Phase 2(RAG)의 성능을 잰다.

**판정 기준(등급, 카테고리, finding 작성 규칙, 경계 사례)은 [`docs/policy.md`](../docs/policy.md)에 있다.** 모델 프롬프트에도 같은 파일이 들어가므로, 기준을 바꿀 때는 policy.md만 고친다.

## 1. 메타 필드

- `label_status`: `draft`(초안) → `reviewed`(2인 검토) → `final`.
- `split`: `dev` / `test`. test는 튜닝하는 동안 보지 않는다. 시드는 모두 `dev`.
- `origin`: 샘플 출처. `synthetic`(가이드라인 예시를 바탕으로 새로 작성), `case_book`(事例解説書 등 사례에서 각색), `real`(실제 등록 데이터, 개인정보·병원명 마스킹).
- `note`: 판정 근거나 애매한 점.

## 2. 목표 분포 (100건 기준)

- `violation` 50%, `caution` 20%, `ok` 30%
- C1·C2·C4~C9 카테고리마다 최소 5건
- 언어: 일본어 70%, 한국어 30%
- 현재 50건: violation 30 / caution 5 / ok 15, 일본어 35 / 한국어 15. **C9 샘플이 3건, caution이 5건으로 아직 부족하다.**

## 3. 검토 절차

1. 라벨러 A가 작성한다 (`draft`).
2. 라벨러 B가 독립적으로 판정하고 차이를 맞춘다 (`reviewed`).
3. 차이가 컸던 경계 사례는 `docs/policy.md` §5 표에 규칙으로 추가한다.
4. 전문가(법률) 확인이 끝나면 `caution` 대기 항목을 다시 판정한다 (`final`).
