from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import stocks, earnings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(earnings.router, prefix="/stocks", tags=["earnings"])
