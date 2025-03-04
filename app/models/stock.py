from typing import List
from pydantic import BaseModel, Field

class StockHistory(BaseModel):
    Open: float = Field(description="Opening price of the stock")
    High: float = Field(description="Highest price of the stock during the period")
    Low: float = Field(description="Lowest price of the stock during the period")
    Close: float = Field(description="Closing price of the stock")
    Volume: float = Field(description="Trading volume")
    Dividends: float = Field(description="Dividend amount")
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