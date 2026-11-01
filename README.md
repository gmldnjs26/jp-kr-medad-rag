# K-Beauty 의료광고 검수 RAG (학습용)

한국 미용 시술을 일본 거주자에게 소개하는 서비스 "K-Beauty"를 위한 RAG. 병원이 관리화면에 시술 제목·설명을 등록하면, 한국·일본 의료광고 규제 위반 소지를 판정하고 수정안을 제시한다.

> 학습용 프로젝트다. 판정 결과는 법률 자문이 아니다.

## 구성

| 경로 | 내용 |
| --- | --- |
| `docs/design.md` | 설계 (GCP 구성, 지식 소스, 판정 카테고리, Phase 계획) |
| `data/raw/{kr,jp,cross}/` | 법령·가이드라인·사례집 원문. 각 폴더 `SOURCES.md`에 출처 URL |
| `data/corpus/core.md` | Phase 1에 통째로 넣는 핵심 자료 묶음 (`src/build_corpus.py`로 생성) |
| `eval/labeling-guide.md` | 정답 판정 기준 |
| `eval/golden.jsonl` | 평가셋 50건 (`eval/golden.md`는 읽기용 렌더링) |
| `src/judge.py` | Gemini 판정 (structured output) |
| `eval/run_eval.py`, `eval/score.py` | 평가 실행·채점 |

## Phase

- **Phase 0**: 라벨링 가이드, 평가셋 50건 ✅
- **Phase 1**: RAG 없이 core corpus(약 18만 토큰)를 통째로 넣는 long-context baseline ✅
- **Phase 2**: 직접 만든 RAG (Cloud SQL + pgvector + pg_bigm, Vertex Ranking API, gemini-embedding)
- **Phase 3**: Vertex AI RAG Engine과 비교

## Phase 1 결과 (gemini-3.8-flash, 50건)

| 지표 | 값 |
| --- | --- |
| verdict 정확도 | 50/50 (100%) |
| 위반 탐지 재현율 | 35/35 (100%) |
| 오탐률 | 0/15 (0%) |
| 문구(span) 재현율 | 91/99 (92%) |
| 근거 조문 일치율 | 83/91 (91%) |

- 처음 채점에서 verdict가 틀린 5건(g016, g020, g036, g039, g040)은 검토 결과 정답 쪽 누락이어서 정답을 보완했다. 그래서 verdict 100%는 이 모델 출력에 맞춰진 면이 있다. 새 샘플(특히 `test` split)로 다시 확인해야 한다.
- 토큰: 입력 914만(99.7% 암묵적 캐시) / 출력 17만. 2026년 단가 기준 약 $1.3.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # GCP 프로젝트 ID 입력
gcloud auth application-default login
.venv/bin/python src/build_corpus.py
.venv/bin/python eval/run_eval.py --limit 3
```

- 모델: global 리전의 `gemini-3.8-flash`. 명시적 context cache는 global 3.x에서 403이 나서, 자료를 매 요청 맨 앞에 두어 암묵적 캐시를 쓴다.
