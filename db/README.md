# 로컬 검색 DB

Postgres 17 + `pgvector` + `pg_bigm`. 설정은 루트의 `docker-compose.yml`, 이미지는 `db/Dockerfile`, 스키마는 `db/init/`.

```bash
docker compose up -d --build   # 빌드 + 기동 (최초 기동 때 db/init/*.sql 실행)
docker compose ps              # 상태 (healthy면 준비 완료)
docker compose exec db psql -U medad -d medad   # SQL 콘솔
docker compose stop            # 정지 (데이터 유지)
docker compose down            # 컨테이너 삭제 (데이터는 볼륨에 남음)
docker compose down -v         # 볼륨까지 삭제 (스키마부터 다시 만든다)
```

접속 정보: `postgresql://medad:medad@localhost:5432/medad` (로컬 전용)
