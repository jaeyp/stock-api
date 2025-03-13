import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class DivergenceResponse(BaseModel):
    ticker: str
    current_price: str
    date: str
    trend_reversal_potential: str  # Final score in range -100 to 100 as a string with 2 decimals
    details: dict

def calculate_rsi(data, window=14):
    # Calculate the Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    # Calculate the Moving Average Convergence Divergence (MACD) and signal line
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    # Calculate Bollinger Bands (upper and lower bands) using rolling SMA and standard deviation
    sma = data['Close'].rolling(window=window).mean()
    rstd = data['Close'].rolling(window=window).std()
    upper_band = sma + (rstd * 2)
    lower_band = sma - (rstd * 2)
    return upper_band, lower_band

def calculate_ichimoku(data):
    # Calculate Ichimoku components
    nine_period_high = data['High'].rolling(window=9).max()
    nine_period_low = data['Low'].rolling(window=9).min()
    data['Tenkan-sen'] = (nine_period_high + nine_period_low) / 2

    period26_high = data['High'].rolling(window=26).max()
    period26_low = data['Low'].rolling(window=26).min()
    data['Kijun-sen'] = (period26_high + period26_low) / 2

    data['Senkou Span A'] = ((data['Tenkan-sen'] + data['Kijun-sen']) / 2).shift(26)
    data['Senkou Span B'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
    data['Chikou Span'] = data['Close'].shift(-26)

def calculate_fibonacci_levels(data):
    # Calculate Fibonacci retracement levels based on the close prices
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    difference = max_price - min_price
    level1 = max_price - difference * 0.236
    level2 = max_price - difference * 0.382
    level3 = max_price - difference * 0.618
    return level1, level2, level3

def calculate_volume_profile(data, bins=20):
    """
    Calculate the volume profile based on closing prices.
    Returns the price level (vp_peak) with the highest total volume,
    along with the histogram and bin edges.
    """
    prices = data['Close'].values
    volumes = data['Volume'].values
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    max_index = np.argmax(hist)
    vp_peak = (bin_edges[max_index] + bin_edges[max_index+1]) / 2
    return vp_peak, hist, bin_edges

def calculate_smi(data, k_period=5, d_period=3, smoothing=3):
    """
    Calculate the Stochastic Momentum Index (SMI) and its signal line (%D).
    The formula applies double exponential smoothing to the price difference and range.
    """
    # Middle price
    M = (data['High'] + data['Low']) / 2
    # Price difference from the middle
    diff = data['Close'] - M
    # Range
    R = data['High'] - data['Low']
    # Double smoothing
    smooth_diff = diff.ewm(span=smoothing, adjust=False).mean().ewm(span=smoothing, adjust=False).mean()
    smooth_R = R.ewm(span=smoothing, adjust=False).mean().ewm(span=smoothing, adjust=False).mean()
    # SMI calculation; avoid division by zero
    smi = 100 * (smooth_diff / (0.5 * smooth_R)).replace([np.inf, -np.inf], 0)
    # %D is the simple moving average of SMI over d_period
    smi_d = smi.rolling(window=d_period).mean()
    return smi, smi_d

def get_stock_data(ticker):
    # Download stock data for the past year with daily intervals
    stock_data = yf.download(ticker, period='1y', interval='1d')
    return stock_data

@router.get("/{ticker}/divergence", response_model=DivergenceResponse)
async def analyze_stock(ticker: str):
    try:
        data = get_stock_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    # Calculate technical indicators
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    calculate_ichimoku(data)
    fib_levels = calculate_fibonacci_levels(data)
    vp_peak, vp_hist, vp_bins = calculate_volume_profile(data, bins=20)
    
    # Calculate SMI and its signal line
    smi, smi_d = calculate_smi(data)
    
    # Check if Bollinger Bands data is available
    if 'Upper Band' not in data or 'Lower Band' not in data:
        raise HTTPException(status_code=400, detail="Bollinger Bands data is missing.")
    if data['Upper Band'].empty or data['Lower Band'].empty:
        raise HTTPException(status_code=400, detail="Bollinger Bands data is empty.")

    # Extract latest values as scalar floats
    latest_rsi = float(data['RSI'].iloc[-1])
    latest_macd = float(data['MACD'].iloc[-1])
    latest_signal = float(data['Signal'].iloc[-1])
    latest_close = float(data['Close'].iloc[-1])
    latest_upper_band = float(data['Upper Band'].iloc[-1])
    latest_lower_band = float(data['Lower Band'].iloc[-1])
    latest_senkou_span_a = float(data['Senkou Span A'].iloc[-1])
    latest_senkou_span_b = float(data['Senkou Span B'].iloc[-1])
    latest_volume = float(data['Volume'].iloc[-1])
    latest_smi = float(smi.iloc[-1])
    latest_smi_d = float(smi_d.iloc[-1])
    
    # Get the previous day's close for volume analysis
    if len(data) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for volume analysis.")
    previous_close = float(data['Close'].iloc[-2])
    
    # Get the latest date from the index
    latest_date = data.index[-1].strftime("%Y-%m-%d")
    
    # ----- Signal Calculations -----
    
    # 1. RSI Signal (proportional; neutral at 50)
    signal_RSI = 50 - latest_rsi
    score_RSI = 2 * signal_RSI

    # 2. MACD Signal (normalized relative to latest close)
    macd_diff = latest_macd - latest_signal
    signal_MACD = (macd_diff / latest_close) * 100  # expressed as percentage
    score_MACD = 10 * signal_MACD

    # 3. Bollinger Bands Signal (continuous proportional signal based on SMA)
    band_width = latest_upper_band - latest_lower_band
    if band_width == 0:
        signal_BB = 0
    else:
        latest_SMA = (latest_upper_band + latest_lower_band) / 2
        signal_BB = (latest_SMA - latest_close) / band_width
    score_BB = 10 * signal_BB

    # 4. Ichimoku Signal (relative difference from the mid-value of Senkou Span A & B)
    ichimoku_mid = (latest_senkou_span_a + latest_senkou_span_b) / 2
    signal_Ichi = (latest_close - ichimoku_mid) / ichimoku_mid
    score_Ichi = 20 * signal_Ichi

    # 5. Divergence Signal (proportional based on RSI deviation from 50 and price change)
    price_change = latest_close - previous_close
    constant_div = 5
    if latest_rsi <= 50:
        signal_div = (50 - latest_rsi) * (price_change / latest_close) * constant_div
    else:
        signal_div = - (latest_rsi - 50) * (price_change / latest_close) * constant_div
    score_div = signal_div  # weight factor 1

    # 6. Volume Profile Signal (relative difference between latest close and vp_peak)
    distance = (latest_close - vp_peak) / vp_peak
    if distance > 0:
        signal_VP = - distance * 100  # bearish if above vp_peak
    elif distance < 0:
        signal_VP = abs(distance) * 100  # bullish if below vp_peak
    else:
        signal_VP = 0
    score_VP = signal_VP

    # 7. Volume Ratio Signal (proportional adjustment based on volume ratio)
    avg_volume = float(data['Volume'].rolling(window=20).mean().iloc[-1])
    volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1
    if latest_close < previous_close:
        signal_Vol = - (volume_ratio - 1) * 10
    elif latest_close > previous_close:
        signal_Vol = (volume_ratio - 1) * 10
    else:
        signal_Vol = 0
    score_Vol = signal_Vol

    # 8. SMI Signal (based on the slopes of K and D over a recent window)
    smi_slope_window = 5
    if len(smi) >= smi_slope_window:
        x = np.arange(smi_slope_window)
        k_values = smi.iloc[-smi_slope_window:].values
        d_values = smi_d.iloc[-smi_slope_window:].values
        k_slope = np.polyfit(x, k_values, 1)[0]
        d_slope = np.polyfit(x, d_values, 1)[0]
        slope_diff = float(k_slope - d_slope)  # Explicitly convert to float
    else:
        slope_diff = 0.0
    score_SMI = 10 * slope_diff

    # ----- Combine All Signals -----
    total_score = score_RSI + score_MACD + score_BB + score_Ichi + score_div + score_VP + score_Vol + score_SMI

    # ----- Normalize Total Score to Range -100 to 100 using tanh -----
    final_score = 100 * np.tanh(total_score / 20)

    # Prepare details dictionary with numbers rounded to 2 decimal places, including SMI K, D and slope_diff
    details = {
        "RSI": {"raw_signal": round(signal_RSI, 2), "weighted_score": round(score_RSI, 2)},
        "MACD": {"raw_signal": round(signal_MACD, 2), "weighted_score": round(score_MACD, 2)},
        "BollingerBands": {"raw_signal": round(signal_BB, 2), "weighted_score": round(score_BB, 2)},
        "Ichimoku": {"raw_signal": round(signal_Ichi, 2), "weighted_score": round(score_Ichi, 2)},
        "Divergence": {"raw_signal": round(signal_div, 2), "weighted_score": round(score_div, 2)},
        "VolumeProfile": {
            "raw_signal": round(signal_VP, 2),
            "weighted_score": round(score_VP, 2),
            "vp_peak": round(vp_peak, 2)
        },
        "VolumeRatio": {"raw_signal": round(signal_Vol, 2), "weighted_score": round(score_Vol, 2)},
        "SMI": {
            "raw_signal": round(slope_diff, 2),
            "weighted_score": round(score_SMI, 2),
            "K": round(latest_smi, 2),
            "D": round(latest_smi_d, 2),
            "slope_diff": round(slope_diff, 2)
        }
    }

    return {
        "ticker": ticker,
        "current_price": f"{latest_close:.2f}",
        "date": latest_date,
        "trend_reversal_potential": f"{final_score:.2f}",
        "details": details
    }
