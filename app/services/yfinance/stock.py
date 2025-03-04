from .base import get_stock_info

def get_stock_history(ticker: str):
    """Get historical stock data."""
    stock, info = get_stock_info(ticker)
    
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

def get_stock_full_info(ticker: str):
    """Get complete stock information from yfinance."""
    stock, info = get_stock_info(ticker)
    return {
        "ticker": ticker,
        "info": info
    } 