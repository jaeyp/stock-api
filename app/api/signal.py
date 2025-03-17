import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Dict, List
#from momentum import analyze, get_stock_data, get_stocks_data
from app.api.momentum import analyze, get_stock_data, get_stocks_data 

router = APIRouter()

# ✅ 안전한 기본 티커 리스트 (default_factory에서 복사하도록 수정)
DEFAULT_TICKERS = ["QLD", "SOXL", "LABU", "FSLR", "ENPH", "PLUG", "BE", "STRL", "BWXT", "OKLO", 
                    "TEM", "RXRX", "CRSP", "ZG", "RDFN", "PGY", "UPST", "HOOD", "ZETA", "S", 
                    "PINS", "U", "LLY", "NVO", "LUNR", "AMZN", "CRM", "UBER"]

class TradeSignalResponse(BaseModel):
    ticker: str
    date: str
    price: str
    strength: Dict[str, str]

class MultiTradeSignalResponse(BaseModel):
    results: List[TradeSignalResponse]

@router.get("/{ticker}/trade_signal", response_model=TradeSignalResponse)
async def get_trade_signal(ticker: str, period: str = '6mo', mode: str = "conservative"):
    try:
        data = get_stock_data(ticker, period)
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
    """ 여러 개의 티커에 대한 트레이드 신호를 반환하는 API """
    print(f"🛠 Request URL: {request.url}")  # 요청 URL 확인
    print(f"✅ Received tickers: {tickers}")

    # ✅ 만약 클라이언트가 tickers를 빈 리스트([])로 넘긴다면, 기본 tickers 사용
    if not tickers:
        tickers = DEFAULT_TICKERS.copy()

    trade_signal_results = []

    try:
        stocks_data = get_stocks_data(tickers, period)
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

            # 'Close', 'High', 'Low', 'Volume' 데이터 추출 및 복사
            data_to_analyze = ticker_data[['Close', 'High', 'Low', 'Volume']].copy()

            conservative_results = analyze(data_to_analyze, "conservative")
            aggressive_results = analyze(data_to_analyze, "aggressive")

            trade_signal_results.append({
                "ticker": ticker,
                "date": conservative_results['date'],
                "price": conservative_results['close'],
                "strength": {
                    "conservative": conservative_results['latest_diff'],
                    "aggressive": aggressive_results['latest_diff']
                }
            })

        except Exception as e:
            print(f"❌ [ERROR] Error processing trade signal for {ticker}: {str(e)}")

    # strength.conservative 값 기준 오름차순 정렬
    trade_signal_results.sort(key=lambda x: float(x["strength"]["conservative"]))
    return {"results": trade_signal_results}
