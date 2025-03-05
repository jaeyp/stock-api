from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import stocks, earnings, options

app = FastAPI(
    title="Stock API",
    description="API for retrieving stock information using yfinance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Include routers
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(earnings.router, prefix="/stocks", tags=["earnings"])
app.include_router(options.router, prefix="/stocks", tags=["options"])

@app.get("/")
async def root():
    return {"message": "Welcome to Stock API"}
