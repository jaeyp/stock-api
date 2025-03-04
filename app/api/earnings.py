from fastapi import APIRouter, HTTPException
from app.models.earnings import EarningsResponse
from app.services.yfinance.earnings import get_stock_earnings

router = APIRouter()

@router.get("/{ticker}/earningsHistory", response_model=EarningsResponse)
async def get_stock_earnings_data(ticker: str):
    try:
        return get_stock_earnings(ticker)
    except Exception as e:
        print(f"Error fetching earnings data: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}") 