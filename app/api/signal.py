import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Dict, List
#from momentum import analyze, get_stock_data, get_stocks_data
from app.api.momentum import analyze, get_stock_data2, get_stocks_data2 

router = APIRouter()

# Safe default ticker list (modified to copy from default_factory)
DEFAULT_TICKERS = ["VOO", "IWF", "IWM", "UWM", "QQQ", "QLD", "TQQQ", "SQQQ" "SOXX", "SMH", "SOXL", "LABU", "BULZ", "DXYZ", "QTUM", 
                   "BITX", "FSLR", "ENPH", "PLUG", "BE", "STRL", "BWXT", "VST","OKLO", "SMR",
                   "TEM", "RXRX", "CRSP", "O", "ZG", "RDFN", "PGY", "UPST", "HOOD", "ZETA", "S", 
                   "PINS", "U", "LLY", "NVO", "LUNR", "AMZN", "CRM", "UBER", "AAPL", "META", "RGTI", "IONQ"]

class TradeSignalResponse(BaseModel):
    ticker: str
    date: str
    price: str
    momentum: str
    strength: Dict[str, str]

class MultiTradeSignalResponse(BaseModel):
    results: List[TradeSignalResponse]

@router.get("/{ticker}/trade_signal", response_model=TradeSignalResponse)
async def get_trade_signal(ticker: str, period: str = '6mo', mode: str = "conservative"):
    try:
        data = get_stock_data2(ticker, period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    conservative_results = analyze(data[['Close', 'High', 'Low', 'Volume']].copy(), "conservative")
    aggressive_results = analyze(data[['Close', 'High', 'Low', 'Volume']].copy(), "aggressive")

    return {
        "ticker": ticker,
        "date": conservative_results['date'],
        "price": conservative_results['close'],
        "momentum": conservative_results['momentum_strength'],
        "strength": {
            "conservative": conservative_results['latest_diff'],
            "aggressive": aggressive_results['latest_diff']
        }
    }    

@router.get("/trade_signals", response_model=MultiTradeSignalResponse)
async def get_trade_signals(
    request: Request,
    tickers: List[str] = Query(default_factory=lambda: DEFAULT_TICKERS.copy(), description="Comma-separated list of stock tickers"),
    period: str = '6mo'
):
    """ API to return trading signals for multiple tickers """
    print(f"Request URL: {request.url}")  # Verify request URL
    print(f"Received tickers: {tickers}")

    # If client sends an empty list ([]), use default tickers
    if not tickers:
        tickers = DEFAULT_TICKERS.copy()

    trade_signal_results = []

    try:
        stocks_data = get_stocks_data2(tickers, period)
    except Exception as e:
        print(f"❌ [ERROR] Error fetching stocks data: {str(e)}")
        return {"results": trade_signal_results}

    is_multi = isinstance(stocks_data.columns, pd.MultiIndex)

    for ticker in tickers:
        try:
            if is_multi:
                if ticker not in stocks_data.columns.levels[0]:
                    print(f"⚠️ [WARNING] No data fetched for {ticker}. Skipping...")
                    continue
                ticker_data = stocks_data[ticker]
            else:
                ticker_data = stocks_data

            if ticker_data.empty:
                print(f"⚠️ [WARNING] No data fetched for {ticker}. Skipping...")
                continue

            # Extract and copy 'Close', 'High', 'Low', 'Volume' data
            data_to_analyze = ticker_data[['Close', 'High', 'Low', 'Volume']].copy()

            conservative_results = analyze(data_to_analyze, "conservative")
            aggressive_results = analyze(data_to_analyze, "aggressive")

            trade_signal_results.append({
                "ticker": ticker,
                "date": conservative_results['date'],
                "price": conservative_results['close'],
                "momentum": conservative_results['momentum_strength'],
                "strength": {
                    "conservative": conservative_results['latest_diff'],
                    "aggressive": aggressive_results['latest_diff']
                }
            })

        except Exception as e:
            print(f"❌ [ERROR] Error processing trade signal for {ticker}: {str(e)}")

    # Sort trade_signal_results by strength.conservative in ascending order
    trade_signal_results.sort(key=lambda x: float(x["strength"]["conservative"]))
    return {"results": trade_signal_results}
