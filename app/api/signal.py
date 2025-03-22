from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from app.api.momentum import analyze, get_stock_data2, get_stocks_data2 

router = APIRouter()

@dataclass
class TickerInfo:
    ticker: str
    name: str
    type: str
    tags: List[str]

DEFAULT_TICKERS: List[TickerInfo] = [
    TickerInfo(ticker="VOO", name="Vanguard S&P 500 ETF", type="ETF", tags=["Index"]),
    TickerInfo(ticker="QQQ", name="Invesco QQQ Trust", type="ETF", tags=["Index", "Growth"]),
    TickerInfo(ticker="IWF", name="iShares Russell 1000 Growth ETF", type="ETF", tags=["Index", "Growth"]),
    TickerInfo(ticker="IWM", name="iShares Russell 2000 ETF", type="ETF", tags=["Index", "Growth"]),
    TickerInfo(ticker="SOXX", name="iShares Semiconductor ETF", type="ETF", tags=["Index", "Growth"]),
    TickerInfo(ticker="SMH", name="VanEck Semiconductor ETF", type="ETF", tags=["Index", "Growth"]),
    TickerInfo(ticker="QLD", name="ProShares Ultra QQQ", type="ETF", tags=["Leveraged", "Index", "Growth"]),
    TickerInfo(ticker="UWM", name="ProShares Ultra Russell2000", type="ETF", tags=["Leveraged", "Index", "Growth"]),
    TickerInfo(ticker="TQQQ", name="ProShares UltraPro QQQ", type="ETF", tags=["Leveraged", "Index", "Growth"]),
    TickerInfo(ticker="SQQQ", name="ProShares UltraPro Short QQQ", type="ETF", tags=["Leveraged", "Alternative"]),
    TickerInfo(ticker="SOXL", name="Direxion Daily Semiconductor Bull 3X Shares", type="ETF", tags=["Leveraged", "Growth"]),
    TickerInfo(ticker="LABU", name="Direxion Daily S&P Biotech Bull 3X Shares", type="ETF", tags=["Leveraged"]),
    TickerInfo(ticker="BULZ", name="MicroSectors Solactive FANG Innovation 3X Leveraged ETNs", type="ETF", tags=["Leveraged", "Growth"]),
    TickerInfo(ticker="GLD", name="SPDR Gold Shares", type="ETF", tags=["Alternative", "Defensive"]),
    TickerInfo(ticker="BITX", name="2x Bitcoin Strategy ETF", type="ETF", tags=["Leveraged", "Alternative"]),
    TickerInfo(ticker="MCHI", name="iShares MSCI China ETF", type="ETF", tags=["Alternative", "Index", "Growth"]),
    TickerInfo(ticker="EFA", name="iShares MSCI EAFE ETF", type="ETF", tags=["Alternative", "Index", "Defensive"]),
    TickerInfo(ticker="DXYZ", name="Destiny Tech100 Inc.", type="ETF", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="QTUM", name="Defiance Quantum ETF", type="ETF", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="FSLR", name="First Solar, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="ENPH", name="Enphase Energy, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="PLUG", name="Plug Power Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="BE", name="Bloom Energy Corporation", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="STRL", name="Sterling Construction Company", type="Stock", tags=["Growth", "Defensive"]),
    TickerInfo(ticker="BWXT", name="BWX Technologies, Inc.", type="Stock", tags=["Dividend", "Defensive"]),
    TickerInfo(ticker="VST", name="Vistra Corp.", type="Stock", tags=["Growth", "Dividend", "Defensive"]),
    TickerInfo(ticker="OKLO", name="Oklo Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="SMR", name="NuScale Power Corporation", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="TEM", name="Tempus AI, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="RXRX", name="Recursion Pharmaceuticals Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="CRSP", name="CRISPR Therapeutics AG", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="O", name="Realty Income Corporation", type="Stock", tags=["Dividend", "Defensive"]),
    TickerInfo(ticker="ZG", name="Zillow Group, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="RDFN", name="Redfin Corporation", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="PGY", name="Pagaya Technologies Ltd.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="UPST", name="Upstart Holdings, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="HOOD", name="Robinhood Markets, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="ZETA", name="Zeta Global Holdings Corp.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="S", name="SentinelOne, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="PINS", name="Pinterest, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="U", name="Unity Software Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="LLY", name="Eli Lilly and Company", type="Stock", tags=["Dividend", "Defensive"]),
    TickerInfo(ticker="NVO", name="Novo Nordisk A/S", type="Stock", tags=["Dividend", "Defensive"]),
    TickerInfo(ticker="LUNR", name="Intuitive Machines, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="AMZN", name="Amazon.com, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="CRM", name="Salesforce, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="UBER", name="Uber Technologies, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="AAPL", name="Apple Inc.", type="Stock", tags=["Dividend", "Growth"]),
    TickerInfo(ticker="META", name="Meta Platforms, Inc.", type="Stock", tags=["Growth"]),
    TickerInfo(ticker="PLTR", name="Palantir Technologies Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="RGTI", name="Rigetti Computing, Inc.", type="Stock", tags=["Growth", "Innovation"]),
    TickerInfo(ticker="IONQ", name="IonQ, Inc.", type="Stock", tags=["Growth", "Innovation"]),
]

def get_ticker_info(symbol: str) -> TickerInfo:
    for info in DEFAULT_TICKERS:
        if info.ticker == symbol:
            return info
    return TickerInfo(ticker=symbol, name="", type="", tags=[])

class TradeSignalResponse(BaseModel):
    ticker: str
    name: str
    type: str
    tags: List[str]
    date: str
    price: str
    momentum: str
    strength: Dict[str, str]

class MultiTradeSignalResponse(BaseModel):
    results: List[TradeSignalResponse]

@router.get("/{ticker}/trade_signal", response_model=TradeSignalResponse)
async def get_trade_signal(ticker: str, period: str = '6mo', mode: str = "conservative"):
    ticker_info = get_ticker_info(ticker)
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
        "name": ticker_info.name,
        "type": ticker_info.type,
        "tags": ticker_info.tags,
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
    tickers: List[str] = Query(default_factory=lambda: [info.ticker for info in DEFAULT_TICKERS], description="Comma-separated list of stock tickers"),
    period: str = '6mo'
):
    print(f"Request URL: {request.url}")
    print(f"Received tickers: {tickers}")
    if not tickers:
        tickers = [info.ticker for info in DEFAULT_TICKERS]
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
            data_to_analyze = ticker_data[['Close', 'High', 'Low', 'Volume']].copy()
            conservative_results = analyze(data_to_analyze, "conservative")
            aggressive_results = analyze(data_to_analyze, "aggressive")
            ticker_info = get_ticker_info(ticker)
            trade_signal_results.append({
                "ticker": ticker,
                "name": ticker_info.name,
                "type": ticker_info.type,
                "tags": ticker_info.tags,
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
    trade_signal_results.sort(key=lambda x: float(x["strength"]["conservative"]))
    return {"results": trade_signal_results}
