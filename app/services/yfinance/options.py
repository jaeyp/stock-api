from typing import Dict, Any, List
import pandas as pd
from .base import get_stock_info

def get_option_chain(ticker: str, expiration_date: str | None = None) -> Dict[str, Any]:
    """Get option chain data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        expiration_date: Optional expiration date. If not provided, uses the first available date.
    
    Returns:
        Dictionary containing option chain data
    """
    try:
        stock, info = get_stock_info(ticker)
        
        # Get available expiration dates
        expiration_dates = stock.options
        
        if not expiration_dates:
            return {
                "ticker": ticker,
                "currentPrice": round(float(info.get("currentPrice", 0)), 2) if info.get("currentPrice") is not None else None,
                "expirationDates": [],
                "optionChain": {
                    "expirationDate": "",
                    "calls": [],
                    "puts": []
                }
            }
        
        # Use provided expiration date or first available date
        target_date = expiration_date if expiration_date in expiration_dates else expiration_dates[0]
        
        # Get option chain data
        opt = stock.option_chain(target_date)
        
        # Debug: Print available columns
        print("Available columns in calls DataFrame:", opt.calls.columns.tolist())
        print("Sample call option data:", opt.calls.iloc[0].to_dict() if not opt.calls.empty else "No call options")
        
        def process_options(options_df: pd.DataFrame) -> List[Dict[str, Any]]:
            if options_df.empty:
                return []
            
            result = []
            for _, row in options_df.iterrows():
                try:
                    option_data = {
                        "strike": round(float(row.get("strike", 0)), 2),
                        "lastPrice": round(float(row.get("lastPrice", 0)), 2) if pd.notna(row.get("lastPrice")) else None,
                        "bid": round(float(row.get("bid", 0)), 2) if pd.notna(row.get("bid")) else None,
                        "ask": round(float(row.get("ask", 0)), 2) if pd.notna(row.get("ask")) else None,
                        "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
                        "openInterest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else None,
                        "impliedVolatility": round(float(row.get("impliedVolatility", 0)), 2) if pd.notna(row.get("impliedVolatility")) else None,
                        "change": round(float(row.get("change", 0)), 2) if pd.notna(row.get("change")) else None,
                        "changePercentage": round(float(row.get("percentChange", 0)), 2) if pd.notna(row.get("percentChange")) else None
                    }
                    result.append(option_data)
                except (ValueError, TypeError) as e:
                    print(f"Error processing option data: {e}")
                    continue
            return result
        
        # Process calls and puts
        calls = process_options(opt.calls)
        puts = process_options(opt.puts)
        
        return {
            "ticker": ticker,
            "currentPrice": round(float(info.get("currentPrice", 0)), 2) if info.get("currentPrice") is not None else None,
            "expirationDates": expiration_dates,
            "optionChain": {
                "expirationDate": target_date,
                "calls": calls,
                "puts": puts
            }
        }
    except Exception as e:
        print(f"Error in get_option_chain: {str(e)}")
        raise 