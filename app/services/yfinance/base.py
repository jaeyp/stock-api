import yfinance as yf
import requests
import time
from fastapi import HTTPException

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def get_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session

def get_stock_info(ticker: str):
    """Get basic stock information from yfinance with custom session and retry logic."""
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    
    retries = 3
    for attempt in range(retries):
        try:
            info = stock.info
            if "longName" not in info:
                raise HTTPException(status_code=404, detail="Stock data not found")
            return stock, info
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                print(f"{ticker} - 429 error occurred. Retrying {attempt+1}/{retries} after waiting.")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                raise HTTPException(status_code=500, detail=error_message)
    raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker} (retried {retries} times)")
