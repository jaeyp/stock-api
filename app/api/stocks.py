from fastapi import APIRouter, HTTPException
from app.models.stock import StockResponse
from app.services.yfinance.stock import get_stock_history

router = APIRouter()

@router.get("/{ticker}", response_model=StockResponse)
async def get_stock_data(ticker: str):
    try:
        return get_stock_history(ticker)
    except Exception as e:
        print(f"Error fetching stock data: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}") 