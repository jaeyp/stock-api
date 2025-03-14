from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api import momentum, stocks, earnings, history, options
from app.api.graph.trade_signal import router as test_router  # test_di

app = FastAPI(
    title="Stock API",
    description="API for retrieving stock information using yfinance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(earnings.router, prefix="/stocks", tags=["earnings"])
app.include_router(history.router, prefix="/stocks", tags=["history"])
app.include_router(options.router, prefix="/stocks", tags=["options"])
app.include_router(momentum.router, prefix="/stocks", tags=["momentum"]) 
app.include_router(test_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Stock API"}

@app.get("/trade_signal", response_class=HTMLResponse)
async def read_test():
    with open("app/templates/trade_signal.html", encoding="utf-8") as f:  # test.html 파일 제공
        return f.read()
