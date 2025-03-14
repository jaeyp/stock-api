from typing import List
from pydantic import BaseModel, Field

class EarningsData(BaseModel):
    date: str = Field(description="Report date")
    revenue: float | None = Field(description="Total revenue")
    earnings: float | None = Field(description="Net earnings")
    eps: float | None = Field(description="Earnings per share")
    reported_eps: float | None = Field(description="Reported earnings per share")
    revenue_forecast: float | None = Field(description="Revenue forecast")
    eps_forecast: float | None = Field(description="EPS forecast")

    class Config:
        title = "EarningsData"
        description = "Quarterly earnings data"

class EarningsResponse(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    company: str = Field(description="Company name")
    earnings_history: List[EarningsData] = Field(description="Historical earnings data")

    class Config:
        title = "EarningsResponse"
        description = "Complete earnings information including historical data"
