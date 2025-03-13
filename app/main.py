from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import stocks, earnings, options, divergence

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
app.include_router(options.router, prefix="/stocks", tags=["options"])
app.include_router(divergence.router, prefix="/stocks", tags=["divergence"]) 

@app.get("/")
async def root():
    return {"message": "Welcome to Stock API"}
