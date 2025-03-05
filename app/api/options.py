from fastapi import APIRouter, HTTPException
from app.models.options import OptionChainResponse
from app.services.yfinance.options import get_option_chain

router = APIRouter()

@router.get("/{ticker}/options", response_model=OptionChainResponse)
async def get_option_chain_data(ticker: str, expiration_date: str | None = None):
    """Get option chain data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        expiration_date: Optional expiration date. If not provided, uses the first available date.
    
    Returns:
        Option chain data including:
        - Current stock price
        - Available expiration dates
        - Call and Put options data for the selected expiration date
          (strike price, last price, bid/ask, volume, open interest, etc.)
    
    Example:
        GET /stocks/AAPL/options
        GET /stocks/AAPL/options?expiration_date=2024-04-19
    """
    try:
        return get_option_chain(ticker, expiration_date)
    except ValueError as e:
        print(f"Value error in option chain data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error fetching option chain data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch option chain data: {str(e)}"
        ) 