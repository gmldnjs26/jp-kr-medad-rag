# 검색 DB: Postgres 17 + pgvector(벡터 검색) + pg_bigm(한·일 키워드 검색)
FROM pgvector/pgvector:pg17

ARG PG_BIGM_VERSION=v1.2-20250903

# pg_bigm은 공식 패키지가 없어 소스에서 빌드한다. 빌드 도구는 설치 후 지운다.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates git postgresql-server-dev-17 \
 && git clone --depth 1 --branch ${PG_BIGM_VERSION} https://github.com/pgbigm/pg_bigm.git /tmp/pg_bigm \
 && make -C /tmp/pg_bigm USE_PGXS=1 \
 && make -C /tmp/pg_bigm USE_PGXS=1 install \
 && rm -rf /tmp/pg_bigm \
 && apt-get purge -y build-essential git postgresql-server-dev-17 \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*
