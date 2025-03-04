from datetime import date
import pandas as pd
from .base import get_stock_info

def get_stock_earnings(ticker: str):
    """Get stock earnings data."""
    stock, info = get_stock_info(ticker)

    # Get earnings dates data
    earnings_dates = stock.earnings_dates
    if earnings_dates is None or earnings_dates.empty:
        return {
            "ticker": ticker,
            "company": info.get("longName", "N/A"),
            "earnings_history": []
        }

    # Get today's date
    today = date.today()

    # Convert earnings data to list of dictionaries
    earnings_data = []
    for timestamp, row in earnings_dates.iterrows():
        # Skip future dates
        if timestamp.date() > today:
            continue
            
        eps = float(row[0]) if not row.empty and not pd.isna(row[0]) else None
        earnings_data.append({
            "date": timestamp.date().strftime("%Y-%m-%d"),
            "revenue": None,
            "earnings": eps * info.get("sharesOutstanding", 0) if eps else None,
            "eps": eps,
            "reported_eps": eps,
            "revenue_forecast": None,
            "eps_forecast": None
        })

    # Sort by date in descending order
    earnings_data.sort(key=lambda x: x["date"], reverse=True)

    return {
        "ticker": ticker,
        "company": info.get("longName", "N/A"),
        "earnings_history": earnings_data
    } 