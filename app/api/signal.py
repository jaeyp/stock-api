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
    favorite: bool

DEFAULT_TICKERS: List[TickerInfo] = [
    TickerInfo(ticker="VOO", name="Vanguard S&P 500 ETF", type="ETF", tags=["Index"], favorite=False),
    TickerInfo(ticker="QQQ", name="Invesco QQQ Trust", type="ETF", tags=["Index", "Growth"], favorite=False),
    TickerInfo(ticker="IWF", name="iShares Russell 1000 Growth ETF", type="ETF", tags=["Index", "Growth"], favorite=False),
    TickerInfo(ticker="IWM", name="iShares Russell 2000 ETF", type="ETF", tags=["Index", "Growth"], favorite=False),
    TickerInfo(ticker="SOXX", name="iShares Semiconductor ETF", type="ETF", tags=["Index", "Growth"], favorite=False),
    TickerInfo(ticker="SMH", name="VanEck Semiconductor ETF", type="ETF", tags=["Index", "Growth"], favorite=False),
    TickerInfo(ticker="QLD", name="ProShares Ultra QQQ", type="ETF", tags=["Leveraged", "Index", "Growth"], favorite=True),
    TickerInfo(ticker="USD", name="ProShares Ultra Semiconductors ", type="ETF", tags=["Leveraged", "Index", "Growth"], favorite=False),
    # 3x leverage ETFs
    TickerInfo(ticker="TQQQ", name="ProShares UltraPro QQQ", type="ETF", tags=["Leveraged", "Index", "Growth"], favorite=False),
    TickerInfo(ticker="SQQQ", name="ProShares UltraPro Short QQQ", type="ETF", tags=["Leveraged", "Alternative"], favorite=False),
    TickerInfo(ticker="SOXL", name="Direxion Daily Semiconductor Bull 3X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="LABU", name="Direxion Daily S&P Biotech Bull 3X Shares", type="ETF", tags=["Leveraged"], favorite=True),
    TickerInfo(ticker="BULZ", name="MicroSectors Solactive FANG Innovation 3X Leveraged ETNs", type="ETF", tags=["Leveraged", "Growth"], favorite=False),
    # 2x leverage ETFs
    TickerInfo(ticker="LLYX", name="Defiance Daily Target 2X Long LLY ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    # GraniteShares
    TickerInfo(ticker="PTIR", name="GraniteShares 2x Long PLTR Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="NVDL", name="GraniteShares 2x Long NVDA Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="AAPB", name="GraniteShares 2x Long AAPL Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="AMZZ", name="GraniteShares 2x Long AMZN Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="MSFL", name="GraniteShares 2x Long MSFT Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="FBL", name="GraniteShares 2x Long META Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="TSLR", name="Graniteshares 2x Long TSLA Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="RDTL", name="GraniteShares 2x Long RDDT Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="CONL", name="GraniteShares 2x Long COIN Daily ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    # Direxion
    #TickerInfo(ticker="ELIL", name="Direxion Daily LLY Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="AVL", name="Direxion Daily AVGO Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    TickerInfo(ticker="GGLL", name="Direxion Daily GOOGL Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="PLTU", name="Direxion Daily PLTR Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="NVDU", name="Direxion Daily Nvda Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="AAPU", name="Direxion Daily AAPL Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="AMZU", name="Direxion Daily AMZN Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="MSFU", name="Direxion Daily MSFT Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="METU", name="Direxion Daily META Bull 2X ETF", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    #TickerInfo(ticker="TSLL", name="Direxion Daily TSLA Bull 2X Shares", type="ETF", tags=["Leveraged", "Growth"], favorite=True),
    # Others
    TickerInfo(ticker="GLD", name="SPDR Gold Shares", type="ETF", tags=["Alternative", "Defensive"], favorite=False),
    TickerInfo(ticker="BITX", name="2x Bitcoin Strategy ETF", type="ETF", tags=["Leveraged", "Alternative"], favorite=False),
    TickerInfo(ticker="KWEB", name="KraneShares CSI China Internet ETF", type="ETF", tags=["Alternative", "Index", "Growth"], favorite=False),
    TickerInfo(ticker="CQQQ", name="Invesco China Technology ETF", type="ETF", tags=["Alternative", "Index", "Growth"], favorite=False),
    TickerInfo(ticker="MCHI", name="iShares MSCI China ETF", type="ETF", tags=["Alternative", "Index", "Growth"], favorite=False),
    TickerInfo(ticker="EWJ", name="iShares MSCI Japan ETF", type="ETF", tags=["Alternative", "Index", "Defensive"], favorite=False),
    TickerInfo(ticker="JPXN", name="iShares JPX-Nikkei 400 ETF", type="ETF", tags=["Alternative", "Index", "Growth"], favorite=False),
    TickerInfo(ticker="EFA", name="iShares MSCI EAFE ETF", type="ETF", tags=["Alternative", "Index", "Defensive"], favorite=False),
    TickerInfo(ticker="DXYZ", name="Destiny Tech100 Inc.", type="ETF", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="QTUM", name="Defiance Quantum ETF", type="ETF", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="FSLR", name="First Solar, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="ENPH", name="Enphase Energy, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="PLUG", name="Plug Power Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="BE", name="Bloom Energy Corporation", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="STRL", name="Sterling Construction Company", type="Stock", tags=["Growth", "Defensive"], favorite=False),
    TickerInfo(ticker="BWXT", name="BWX Technologies, Inc.", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="VST", name="Vistra Corp.", type="Stock", tags=["Growth", "Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="OKLO", name="Oklo Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="SMR", name="NuScale Power Corporation", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="TEM", name="Tempus AI, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="RXRX", name="Recursion Pharmaceuticals Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="CRSP", name="CRISPR Therapeutics AG", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="O", name="Realty Income Corporation", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="ZG", name="Zillow Group, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="RDFN", name="Redfin Corporation", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="PGY", name="Pagaya Technologies Ltd.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="UPST", name="Upstart Holdings, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="HOOD", name="Robinhood Markets, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="ZETA", name="Zeta Global Holdings Corp.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="S", name="SentinelOne, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="PINS", name="Pinterest, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="U", name="Unity Software Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="LLY", name="Eli Lilly and Company", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="NVO", name="Novo Nordisk A/S", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="LUNR", name="Intuitive Machines, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=True),
    TickerInfo(ticker="CRM", name="Salesforce, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="UBER", name="Uber Technologies, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="MSFT", name="Microsoft Corporation", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="AMZN", name="Amazon.com, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="GOOG", name="Alphabet Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="AAPL", name="Apple Inc.", type="Stock", tags=["Dividend", "Growth"], favorite=False),
    TickerInfo(ticker="META", name="Meta Platforms, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="NVDA", name="NVIDIA Corporation", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="AVGO", name="Broadcom Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="TSLA", name="Tesla, Inc.", type="Stock", tags=["Growth"], favorite=False),
    TickerInfo(ticker="PLTR", name="Palantir Technologies Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="RGTI", name="Rigetti Computing, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="IONQ", name="IonQ, Inc.", type="Stock", tags=["Growth", "Innovation"], favorite=False),
    TickerInfo(ticker="MCD", name="McDonald's Corporation", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="WMT", name="Walmart Inc.", type="Stock", tags=["Dividend", "Defensive"], favorite=False),
    TickerInfo(ticker="OLLI", name="Ollie's Bargain Outlet Holdings, Inc.", type="Stock", tags=["Defensive"], favorite=False),
]

def get_ticker_info(symbol: str) -> TickerInfo:
    for info in DEFAULT_TICKERS:
        if info.ticker == symbol:
            return info
    return TickerInfo(ticker=symbol, name="", type="", tags=[], favorite=False)

class TradeSignalResponse(BaseModel):
    ticker: str
    name: str
    type: str
    tags: List[str]
    favorite: bool
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
        "favorite": ticker_info.favorite,
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
                    continue
                ticker_data = stocks_data[ticker]
            else:
                ticker_data = stocks_data
                
            if ticker_data.empty:
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
                "favorite": ticker_info.favorite,
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
