# jp-kr-medad-rag: 한·일 의료광고 검수 RAG (학습용)

한국 미용 시술을 일본 거주자에게 소개하는 **가상의 중개 플랫폼**을 전제로 한 RAG. 병원이 관리화면에 시술 제목·설명을 등록하면, 한국·일본 의료광고 규제 위반 소지를 판정하고 수정안을 제시한다.

> 학습용 프로젝트다. 판정 결과는 법률 자문이 아니다.

## 진행 상황

| Phase | 내용 | 상태 | 기록 |
| --- | --- | --- | --- |
| 0 | 판정 기준, 평가셋 50건 → 223건 | ✅ | [`policy.md`](docs/policy.md), [`labeling-guide.md`](eval/labeling-guide.md) |
| 1 | RAG 없이 자료 전체를 넣는 long-context baseline | ✅ | [`phase1-log.md`](docs/phase1-log.md) |
| 2 | 직접 만든 RAG (청킹, multi-vector, 질의 바꿔 쓰기, RRF, pgvector) | 검색·판정 ✅, 하이브리드·rerank 진행 중 | [`phase2-retrieval-log.md`](docs/phase2-retrieval-log.md) |
| 3 | Vertex AI RAG Engine과 비교 | 예정 | |

## 핵심 결과 (2026-11-08, 223건, `gemini-3.8-flash`)

| | RAG top-10 | 자료 전체 (약 32만 토큰) |
| --- | --- | --- |
| verdict 정확도 | 90% | 91% |
| 문구 재현율 / 적합률 | 88% / 85% | 94% / 81% |
| 근거 조문 일치율 | 73% | 85% |
| 입력 토큰 (1건) | 약 9.4천 | 약 33만 |
| 추정 월 비용 (하루 100건) | 약 $40 | 약 $320 |

- 검색 재현율 recall@10 74%, recall@20 86% (RRF).
- 최종 등급은 거의 같고, 근거 조문은 자료 전체가 정확하다. 비용 추정의 가정과 근거는 [기록 9절](docs/phase2-retrieval-log.md#9-rag-판정-vs-자료-전체-판정-2026-11-08), 지표의 뜻은 [`metrics.md`](docs/metrics.md).

## 구성

| 경로 | 내용 |
| --- | --- |
| `docs/design.md` | 설계 (GCP 구성, 지식 소스, 판정 카테고리) |
| `docs/policy.md` | 판정 기준. 모델 프롬프트에 매번 넣고, 정답을 만들 때도 쓴다 |
| `docker-compose.yml`, `db/` | 로컬 검색 DB (Postgres + pgvector + pg_bigm) |
| `data/raw/{kr,jp}/` | 검색 대상 원문 13개 (2026-11-07 재선정) |
| `data/chunks/` | 청크(`chunks.jsonl`), 청크별 광고 예시(`examples_v3.jsonl`) |
| `data/queries/` | 질의 바꿔 쓰기 결과 |
| `data/corpus/` | 통째로 넣는 자료. `core_v2.md`가 현재 자료, `core.md`는 Phase 1 당시 자료 |
| `eval/golden.jsonl` | 평가셋 223건 (`golden.md`는 읽기용 렌더링) |
| `src/chunk.py` → `augment.py` → `embed.py` | 청킹 → 청크별 광고 예시 → 임베딩 |
| `src/load_db.py` | 청크와 벡터를 DB에 적재 |
| `src/rewrite.py`, `src/retrieve.py` | 질의 바꿔 쓰기, RRF 검색 (pgvector) |
| `src/judge.py` | Gemini 판정 (`judge`: 자료 전체, `judge_rag`: 검색 결과만) |
| `eval/retrieval.py`, `eval/run_eval.py`, `eval/score.py` | 검색 평가, 판정 실행, 채점 |

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # GCP 프로젝트 ID 입력
gcloud auth application-default login

.venv/bin/python src/chunk.py                         # 청킹 → data/chunks/chunks.jsonl
.venv/bin/python src/augment.py --out examples_v3 --variant v2c   # 청크별 광고 예시 (LLM)
.venv/bin/python src/rewrite.py                       # 평가셋 질의 바꿔 쓰기 (LLM)
.venv/bin/python src/embed.py && .venv/bin/python src/embed.py --examples examples_v3
docker compose up -d --build && .venv/bin/python src/load_db.py   # DB 기동 + 적재

.venv/bin/python eval/retrieval.py --fuse examples_v3              # 검색 재현율
.venv/bin/python eval/run_eval.py --rag 10                         # RAG 판정
.venv/bin/python eval/run_eval.py --corpus data/corpus/core_v2.md  # 자료 전체 판정
```

- 모델은 global 리전의 `gemini-3.8-flash`. 자료 전체 판정은 명시적 context cache를 먼저 시도하고, 실패하면 자료를 매 요청 맨 앞에 두어 암묵적 캐시를 쓴다.
