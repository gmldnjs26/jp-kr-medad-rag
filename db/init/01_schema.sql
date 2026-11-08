-- 최초 기동 때 한 번 실행된다. 다시 적용하려면 `docker compose down -v` 후 재기동.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_bigm;

-- data/chunks/chunks.jsonl 한 줄 = 한 행
CREATE TABLE chunks (
    id          text PRIMARY KEY,   -- c00000 ...
    country     text NOT NULL,      -- JP | KR
    doc         text NOT NULL,
    ref         text NOT NULL,
    source_file text NOT NULL,
    page        int,
    pages       int[],
    text        text NOT NULL
);

-- multi-vector: 청크 하나에 벡터 여러 개. 검색은 chunk_id별 최대 유사도를 쓴다.
CREATE TABLE chunk_vectors (
    chunk_id  text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    kind      text NOT NULL,        -- orig: 원문, example: 광고 예시 (예시가 없는 청크는 행 없음)
    model     text NOT NULL,        -- gemini-embedding-001_768
    embedding vector(768) NOT NULL,
    PRIMARY KEY (chunk_id, kind, model)
);
