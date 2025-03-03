from fastapi import FastAPI, HTTPException
import yfinance as yf
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note on naming convention:
# While Python typically uses snake_case, the field names in StockHistory follow PascalCase
# to maintain consistency with the original yfinance library's data structure.
# This decision prioritizes maintaining the external library's convention over Python's standard.
class StockHistory(BaseModel):
    Open: float = Field(description="Opening price of the stock")
    High: float = Field(description="Highest price of the stock during the period")
    Low: float = Field(description="Lowest price of the stock during the period")
    Close: float = Field(description="Closing price of the stock")
    Volume: float = Field(description="Trading volume")
    Dividends: float = Field(description="Dividend amount")
    # Originally "Stock Splits" in yfinance data.
    # Renamed to "StockSplits" to:
    # 1. Remove spaces for better code handling
    # 2. Maintain PascalCase consistency with other fields
    # 3. Ensure compatibility with TypeScript/JavaScript property access
    StockSplits: float = Field(description="Stock split ratio")
    Date: str | None = Field(default=None, description="Date of the stock data")

    class Config:
        title = "StockHistory"
        description = "Historical stock data for a single day"

class StockResponse(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    company: str = Field(description="Company name")
    current_price: float | None = Field(description="Current stock price")
    market_cap: float | None = Field(description="Market capitalization")
    history: List[StockHistory] = Field(description="Historical stock data")

    class Config:
        title = "StockResponse"
        description = "Complete stock information including historical data"

@app.get("/stocks/{ticker}", response_model=StockResponse)
async def get_stock_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)

        # Get company information
        info = stock.info
        if "longName" not in info:
            raise HTTPException(status_code=404, detail="Stock data not found")

        # Get 5-day historical data
        history = stock.history(period="5d")
        
        # Convert DataFrame to list of dictionaries and add date information
        history_data = []
        for date, row in history.iterrows():
            data = row.to_dict()
            # Convert "Stock Splits" to "StockSplits"
            if "Stock Splits" in data:
                data["StockSplits"] = data.pop("Stock Splits")
            # Add date information (ISO format)
            data["Date"] = date.isoformat()[:10]  # YYYY-MM-DD format
            history_data.append(data)

        return {
            "ticker": ticker,
            "company": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "history": history_data
        }

    except Exception as e:
        print(f"Error fetching stock data: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
