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

-- 청크별 광고 예시 문장 (data/chunks/examples_v3.jsonl). 키워드 검색 C1에서 광고 문구와 비교한다
CREATE TABLE chunk_examples (
    chunk_id text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    idx      int  NOT NULL,
    text     text NOT NULL,
    PRIMARY KEY (chunk_id, idx)
);

-- pg_bigm 인덱스: 2글자 단위로 색인해 LIKE '%단어%'와 유사도 검색을 빠르게 한다
CREATE INDEX chunks_text_bigm ON chunks USING gin (text gin_bigm_ops);
CREATE INDEX chunk_examples_text_bigm ON chunk_examples USING gin (text gin_bigm_ops);
