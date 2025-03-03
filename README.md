# stock-api

## Distribution

api server (w/ PostgreSQL+Redis hybrid): railway (free 500hours/month) -> fly.io (docker base, free w/ 128MB RAM limit)
[FastAPI] → [Redis 캐시 확인] → [캐시 HIT] → 응답 반환 ✅
                            ↘ [캐시 MISS] → [PostgreSQL 조회 후 Redis 저장] → 응답 반환
vue app: vercel
flutter app: vercel
