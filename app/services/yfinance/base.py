import yfinance as yf
from fastapi import HTTPException

def get_stock_info(ticker: str):
    """Get basic stock information from yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    if "longName" not in info:
        raise HTTPException(status_code=404, detail="Stock data not found")
    return stock, info 