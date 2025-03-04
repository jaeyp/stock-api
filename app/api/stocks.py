from fastapi import APIRouter, HTTPException
from app.models.stock import StockResponse, StockInfo
from app.services.yfinance.stock import get_stock_history, get_stock_full_info

router = APIRouter()

@router.get("/{ticker}", response_model=StockResponse)
async def get_stock_data(ticker: str):
    try:
        return get_stock_history(ticker)
    except Exception as e:
        print(f"Error fetching stock data: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/{ticker}/info", response_model=StockInfo)
async def get_stock_full_data(ticker: str):
    try:
        return get_stock_full_info(ticker)
    except Exception as e:
        print(f"Error fetching stock info: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}") 