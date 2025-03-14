from typing import List
from pydantic import BaseModel, Field

class HistoryData(BaseModel):
    Date: str = Field(description="Date of the stock price")
    Open: float = Field(description="Opening price")
    High: float = Field(description="Highest price")
    Low: float = Field(description="Lowest price")
    Close: float = Field(description="Closing price")
    Volume: int = Field(description="Trading volume")

    class Config:
        title = "HistoryData"
        description = "daily stock price data"

class HistoryResponse(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    history: List[HistoryData] = Field(description="List of historical stock price data")

    class Config:
        title = "HistoryResponse"
        description = "Historical stock price data including date, open, high, low, close, and volume" 