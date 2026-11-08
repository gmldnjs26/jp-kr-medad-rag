# jp-kr-medad-rag: 한·일 의료광고 검수 RAG (학습용)

한국 미용 시술을 일본 거주자에게 소개하는 **가상의 중개 플랫폼**을 전제로 한 RAG. 병원이 관리화면에 시술 제목·설명을 등록하면, 한국·일본 의료광고 규제 위반 소지를 판정하고 수정안을 제시한다.

> 학습용 프로젝트다. 판정 결과는 법률 자문이 아니다.

## 구성

| 경로 | 내용 |
| --- | --- |
| `docs/design.md` | 설계 (GCP 구성, 지식 소스, 판정 카테고리, Phase 계획) |
| `data/raw/{kr,jp}/` | 검색 대상 원문 13개 (2026-11-07 재선정). 폴더 `SOURCES.md`의 일부 항목은 삭제 전 자료 기준이다 |
| `data/corpus/core.md` | Phase 1에 통째로 넣은 자료 묶음. 삭제 전 자료로 만든 것이라 지금 원문으로는 다시 만들 수 없다 |
| `docs/policy.md` | 판정 기준. 모델 프롬프트에 매번 넣고, 정답을 만들 때도 쓴다 |
| `eval/labeling-guide.md` | 정답 작성 절차 (메타 필드, 목표 분포, 검토 절차) |
| `eval/golden.jsonl` | 평가셋 223건 (g001~g050 Phase 0, g051~ 새 자료 기반). finding마다 검색 근거 `evidence`. `eval/golden.md`는 읽기용 렌더링 |
| `src/judge.py` | Gemini 판정 (structured output) |
| `eval/run_eval.py`, `eval/score.py` | 평가 실행·채점 |
| `src/chunk.py`, `src/embed.py`, `src/augment.py`, `src/rewrite.py` | Phase 2: 청킹, 임베딩, 청크별 광고 예시(multi-vector), 질의 바꿔 쓰기 |
| `eval/retrieval.py` | Phase 2 검색 재현율 (`--fuse`: RRF) |
| `docs/phase2-retrieval-log.md` | Phase 2 검색 개선 기록 |

## Phase

- **Phase 0**: 라벨링 가이드, 평가셋 50건 ✅ (2026-11-07에 223건으로 확장)
- **Phase 1**: RAG 없이 core corpus(약 18만 토큰)를 통째로 넣는 long-context baseline ✅
- **Phase 2**: 직접 만든 RAG. 검색까지 진행 중이다 (RRF 기준 recall@10 74%, 기록은 `docs/phase2-retrieval-log.md`). 저장소는 아직 로컬 numpy
- **Phase 3**: Vertex AI RAG Engine과 비교

## Phase 1 결과 (gemini-3.8-flash, 50건)

| 지표 | 값 |
| --- | --- |
| verdict 정확도 | 50/50 (100%) |
| 위반 탐지 재현율 | 35/35 (100%) |
| 오탐률 | 0/15 (0%) |
| 문구(span) 재현율 | 124/133 (93%) |
| 문구(span) 적합률 | 130/148 (88%) |
| 근거 조문 일치율 | 103/124 (83%), 법령명과 항·호까지 비교 |
| C8(누락) 재현율 / 적합률 | 44/50 (88%) / 44/46 (96%), 샘플×나라 단위 |

- 처음 채점에서 verdict가 틀린 5건(g016, g020, g036, g039, g040)은 검토 결과 정답 쪽 누락이어서 정답을 보완했다. 그래서 verdict 100%는 이 모델 출력에 맞춰진 면이 있다. 새 샘플(특히 `test` split)로 다시 확인해야 한다.
- 2026-11-01 정답 보완: 판정 기준(`docs/policy.md`) §1·§4(같은 표현은 양국 각각 finding)와 §5(C8)를 50건 전체에 일괄 적용해 finding 71건을 추가했다. 모델 출력이 아니라 가이드 규칙에서 출발했지만, 보완 전 적합률(67%)이 낮은 원인을 모델 출력에서 발견했으므로 같은 주의가 필요하다.
- 토큰: 입력 914만(99.7% 암묵적 캐시) / 출력 17만. 2026년 단가 기준 약 $1.3.

### 지표 읽는 법

- **정답**은 `eval/golden.jsonl`에 미리 써 둔 답안이다. 샘플마다 `verdict`(최종 등급)와 `findings`(지적 목록)가 있다. finding 하나는 「어느 문구(`span`)가 어느 나라(`country`)의 어느 조문(`rule`)에 걸리는가」이다. 비용·리스크 누락(C8)은 짚을 문구가 없어 `span`이 비어 있다.
- **모델 출력**도 같은 형식이다. `src/judge.py`가 structured output(`response_schema=Judgment`)으로 `verdict`와 `findings`를 JSON으로 받는다.
- 샘플 단위 지표:
  - verdict 정확도: 최종 등급이 정답과 같은 비율
  - 위반 탐지 재현율: 정답이 `violation`·`caution`인 샘플 중 모델도 `ok`가 아니라고 본 비율
  - 오탐률: 정답이 `ok`인 샘플 중 모델이 문제 있다고 본 비율
- finding 단위 지표:
  - **재현율**: 정답 findings 중 모델 findings와 일치하는 비율. 찾아야 할 것을 얼마나 빠짐없이 찾았나. 놓치면 내려간다.
  - **적합률**: 모델 findings 중 정답 findings와 일치하는 비율. 찾았다고 낸 것 중 얼마나 맞았나. 헛짚으면 내려간다.
  - 「일치」는 같은 샘플·같은 나라에서 span이 겹치는 것이다. 한쪽이 다른 쪽을 포함하거나, 공통 부분이 짧은 쪽의 절반 이상이면 일치로 본다(`eval/score.py`의 `span_match`). 일치가 1:1이 아니어서 재현율과 적합률의 분자가 다를 수 있다.
  - 근거 조문 일치율: span이 일치한 finding 중 `rule`의 법령명과 조·항·호까지 맞은 비율
  - C8은 span이 없으므로 (샘플, 나라) 쌍으로 재현율·적합률을 센다.
- 재현율과 적합률은 반대로 움직이기 쉽다. 의심 가는 문구를 다 짚으면 재현율은 오르고 적합률은 내려간다. 검수 도구에서는 재현율을 더 중요하게 본다. 놓친 위반은 그대로 게시되지만, 헛짚은 지적은 사람이 검토하면서 걸러낼 수 있다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # GCP 프로젝트 ID 입력
gcloud auth application-default login
.venv/bin/python src/build_corpus.py
.venv/bin/python eval/run_eval.py --limit 3
```

- 모델: global 리전의 `gemini-3.8-flash`. 명시적 context cache는 global 3.x에서 403이 나서, 자료를 매 요청 맨 앞에 두어 암묵적 캐시를 쓴다.
