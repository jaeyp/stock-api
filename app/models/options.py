from typing import List, Dict, Any
from pydantic import BaseModel, Field

class OptionData(BaseModel):
    strike: float = Field(description="Strike price of the option")
    lastPrice: float | None = Field(description="Last traded price")
    bid: float | None = Field(description="Current bid price")
    ask: float | None = Field(description="Current ask price")
    volume: int | None = Field(description="Trading volume")
    openInterest: int | None = Field(description="Open interest")
    impliedVolatility: float | None = Field(description="Implied volatility")
    change: float | None = Field(description="Price change from previous close")
    changePercentage: float | None = Field(description="Percentage change from previous close")

    class Config:
        title = "OptionData"
        description = "Data for a single option contract"

class OptionChain(BaseModel):
    expirationDate: str = Field(description="Option expiration date")
    calls: List[OptionData] = Field(description="List of call options")
    puts: List[OptionData] = Field(description="List of put options")

    class Config:
        title = "OptionChain"
        description = "Complete option chain data for a specific expiration date"

class OptionChainResponse(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    currentPrice: float | None = Field(description="Current stock price")
    expirationDates: List[str] = Field(description="Available expiration dates")
    optionChain: OptionChain = Field(description="Option chain data for the selected expiration date")

    class Config:
        title = "OptionChainResponse"
        description = "Complete option chain information" 