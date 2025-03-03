# stock-api

*Distribution Plan*

api server (w/ PostgreSQL+Redis hybrid): railway (free 500hours/month) -> fly.io (docker base, free w/ 128MB RAM limit)  
[FastAPI] → [Check Redis cache] → [Cache HIT] → Return response  
　　　　　　　　　↘ [Cache MISS] → [Query PostgreSQL, store in Redis] → Return response  
vue app: vercel  
flutter app: vercel

