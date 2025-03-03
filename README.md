# stock-api

**Distribution Plan**

- API Server (w/ PostgreSQL+Redis hybrid):  
*Railway* (free 500hours/month) -> *Fly.io* (docker base, free w/ 128MB RAM limit)  
[FastAPI] → [Check Redis cache] → [Cache HIT] → Return response  
　　　　　　　　　↘ [Cache MISS] → [Query PostgreSQL, store in Redis] → Return response  
- Vue App: *Vercel*  
- Flutter App: *Vercel*  

