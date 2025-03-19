import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from pytz import timezone
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

router = APIRouter()

# Individual weights for each indicator (adjust as needed)
# Select aggressive/conservative trading mode (default: conservative)
def get_weight_set(mode: str):
    if mode == "aggressive":
        return {
            "RSI": 2,
            "MACD": 10,
            "BB": 10,
            "ICHIMOKU": 20,
            "DIV": 1,
            "VP": 1,
            "VOL": 2,
            "SMI": 2,
            "FIB": 0,   # exclude FIB 4,
        }
    else:  # Default: conservative
        return {
            "RSI": 1,
            "MACD":  1,
            "BB": 10,
            "ICHIMOKU": 20,
            "DIV": 1,
            "VP": 1,
            "VOL": 2,
            "SMI": 2,
            "FIB": 0,   # exclude FIB 1,
        }
    
# Configurable slope window sizes
MACD_SLOPE_WINDOW = 4
SMI_SLOPE_WINDOW = 2

# Default settings
USE_NORMALIZE_MOMENTUM_STRENGTH = False
USE_MACD_SLOPE = True

class MomentumStrengthResponse(BaseModel):
    ticker: str
    history: dict  # Date -> trend score

class MomentumResponse(BaseModel):
    ticker: str
    date: str
    current_price: str
    scaled_price: str
    momentum_strength: str  # Final score in range -100 to 100 (string, 2 decimals)
    latest_diff: str
    details: dict

def calculate_rsi(data, window=14):
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index), pd.Series([np.nan] * len(data), index=data.index)

    sma = data['Close'].rolling(window=window).mean()
    rstd = data['Close'].rolling(window=window).std()
    upper_band = sma + (rstd * 2)
    lower_band = sma - (rstd * 2)
    return upper_band, lower_band

def calculate_ichimoku(data):
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
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    
    # For a Series, convert it to a scalar value.
    if hasattr(max_price, 'item'):
        max_price = max_price.item()
    if hasattr(min_price, 'item'):
        min_price = min_price.item()
    
    difference = max_price - min_price
    levels = {
        "level_23.6": max_price - difference * 0.236,
        "level_38.2": max_price - difference * 0.382,
        "level_50": max_price - difference * 0.5,
        "level_61.8": max_price - difference * 0.618,
        "level_78.6": max_price - difference * 0.786
    }
    return levels

def weighted_median(values, weights):
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cutoff = cumulative_weights[-1] / 2.0
    median_index = np.searchsorted(cumulative_weights, cutoff)
    return float(sorted_values[median_index])

def calculate_volume_profile(data, bins=20):
    prices = data['Close'].values
    volumes = data['Volume'].values
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    max_index = np.argmax(hist)
    vp_peak = (bin_edges[max_index] + bin_edges[max_index+1]) / 2
    vp_median = weighted_median(prices, volumes)
    return vp_peak, vp_median, hist, bin_edges

def calculate_smi(data, k_period=10, k_smoothing=3, k_double_smoothing=3, d_period=10):
    # 각 시점에서 최근 k_period 동안의 최고가와 최저가를 계산합니다.
    highest_high = data['High'].rolling(window=k_period, min_periods=1).max()
    lowest_low = data['Low'].rolling(window=k_period, min_periods=1).min()
    
    # 중앙값 계산
    median = (highest_high + lowest_low) / 2
    
    # 원시 값과 범위 계산
    raw = data['Close'] - median
    raw_range = (highest_high - lowest_low) / 2
    
    # 첫 번째 지수이동평균 (스무딩)
    smooth_raw = raw.ewm(span=k_smoothing, adjust=False).mean()
    smooth_range = raw_range.ewm(span=k_smoothing, adjust=False).mean()
    
    # 두 번째 지수이동평균 (이중 스무딩)
    double_smooth_raw = smooth_raw.ewm(span=k_double_smoothing, adjust=False).mean()
    double_smooth_range = smooth_range.ewm(span=k_double_smoothing, adjust=False).mean()
    
    # 분모가 0인 경우를 방지
    double_smooth_range = double_smooth_range.replace(0, np.nan)
    
    # SMI 계산
    smi = 100 * (double_smooth_raw / double_smooth_range)
    
    # %D는 SMI의 지수이동평균 (d_period)
    smi_d = smi.ewm(span=d_period, adjust=False).mean()
    
    # 결측치는 0으로 채워줍니다.
    smi = smi.fillna(0)
    smi_d = smi_d.fillna(0)
    
    return smi, smi_d

def compute_divergence_series(data):
    divergences = []
    for i in range(1, len(data)):
        rsi_val = data['RSI'].iloc[i]
        price_change = data['Close'].iloc[i] - data['Close'].iloc[i-1]
        close_val = data['Close'].iloc[i]
        if rsi_val <= 50:
            div_val = (50 - rsi_val) * (price_change / close_val) * 5
        else:
            div_val = - (rsi_val - 50) * (price_change / close_val) * 5
        divergences.append(div_val)
    return pd.Series(divergences, index=data.index[1:])

def normalize_scores_tanh(momentum_strength):
    scores = np.array(list(momentum_strength.values()))
    
    mean = scores.mean()
    std = scores.std()

    if std == 0:
        normalized_scores = np.zeros_like(scores)
    else:
        standardized_scores = (scores - mean) / std
        normalized_scores = 100 * np.tanh(standardized_scores)

    return dict(zip(momentum_strength.keys(), normalized_scores))

def calculate_final_score(indicators):
    # Add only non-NaN scores to valid_scores
    valid_scores = [score for raw, score in indicators if not np.isnan(raw)]
    if valid_scores:
        return round(sum(valid_scores), 2)
    else:
        print("❌ [ERROR] Unable to calculate final_score: All indicators are NaN!")
        return None

def get_stock_data(ticker, period='1y'):
    stock_data = yf.download(ticker, period=period, interval='1d', auto_adjust=False)
    return stock_data

def get_stock_data2(ticker, period='1y', reference_date=None):
    """
    Downloads historical stock data from a given reference_date for a specified period.
    
    Parameters:
      ticker (str): The stock ticker.
      period (str): A string representing the period (e.g., '1y', '6mo', '30d').
      reference_date (datetime or str): The reference end date for data.
      
    Returns:
      DataFrame: Historical stock data between the calculated start date and the reference_date.
    """
    tz = timezone('Canada/Mountain')

    if reference_date is None:
        reference_date = datetime.now(tz)
    elif not isinstance(reference_date, datetime):
        # Convert string or other formats to datetime
        reference_date = pd.to_datetime(reference_date).to_pydatetime()
        reference_date = reference_date.astimezone(tz)

    period = period.lower().strip()
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = reference_date - relativedelta(years=years)
    elif period.endswith('mo'):
        months = int(period[:-2])
        start_date = reference_date - relativedelta(months=months)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_date = reference_date - timedelta(days=days)
    else:
        raise ValueError("Unsupported period format. Use '1y', '6mo', '30d', etc.")

    start = start_date.strftime("%Y-%m-%d")
    end = (reference_date + timedelta(days=1)).strftime("%Y-%m-%d") # Set the end date to the day after the reference_date

    print('get_stock_data2', start, end)
    stock_data = yf.download(
        ticker,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=False
    )
    return stock_data

def get_stocks_data(tickers: List[str], period: str = '1y') -> pd.DataFrame:
    stocks_data = yf.download(tickers, period=period, interval='1d', auto_adjust=False, group_by='ticker')
    return stocks_data

def get_stocks_data2(tickers: List[str], period: str = '1y', reference_date=None) -> pd.DataFrame:
    tz = timezone('Canada/Mountain')

    if reference_date is None:
        reference_date = datetime.now(tz)
    elif not isinstance(reference_date, datetime):
        # Convert string or other formats to datetime
        reference_date = pd.to_datetime(reference_date).to_pydatetime()
        reference_date = reference_date.astimezone(tz)

    period = period.lower().strip()
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = reference_date - relativedelta(years=years)
    elif period.endswith('mo'):
        months = int(period[:-2])
        start_date = reference_date - relativedelta(months=months)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_date = reference_date - timedelta(days=days)
    else:
        raise ValueError("Unsupported period format. Use '1y', '6mo', '30d', etc.")

    start = start_date.strftime("%Y-%m-%d")
    end = (reference_date + timedelta(days=1)).strftime("%Y-%m-%d") # Set the end date to the day after the reference_date

    stocks_data = yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=False,
        group_by='ticker'
    )
    return stocks_data

def analyze(data, mode="conservative", reference_date=None):
    # 기존 단일 시점 분석 (현재 시점 전용)
    weights = get_weight_set(mode)  # Set weights based on trading style
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    calculate_ichimoku(data)
    fib_levels = calculate_fibonacci_levels(data)
    vp_peak, vp_median, vp_hist, vp_bins = calculate_volume_profile(data, bins=20)
    smi, smi_d = calculate_smi(data)
    div_series = compute_divergence_series(data)
    div_series = pd.to_numeric(div_series, errors='coerce')

    print(f"⚠️ [DEBUG] Mode: {mode}")
    print(f"⚠️ [DEBUG] Data Shape: {data.shape}")
    print(f"⚠️ [DEBUG] Data Head:\n{data.head()}")
    print(f"⚠️ [DEBUG] RSI NaN Count: {data['RSI'].isna().sum()}")
    print(f"⚠️ [DEBUG] MACD NaN Count: {data['MACD'].isna().sum()}")
    print(f"⚠️ [DEBUG] Bollinger Bands NaN Count: {data['Upper Band'].isna().sum()}")
    print(f"⚠️ [DEBUG] Ichimoku NaN Count: {data['Senkou Span A'].isna().sum()}")

    if data['Upper Band'].empty or data['Lower Band'].empty:
        raise HTTPException(status_code=400, detail="Bollinger Bands data is empty.")

    # Extract latest values using .iloc for single element access
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
    
    if len(data) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for volume analysis.")
    previous_close = float(data['Close'].iloc[-2])
    latest_date = data.index[-1].strftime("%Y-%m-%d")
    
    # ----- Raw Indicator Signals -----
    # 1. RSI: deviation from 50
    raw_RSI = 40 - latest_rsi
    score_RSI = weights['RSI'] * raw_RSI
    
    # 2. MACD: using slopes over recent window (with MACD_SLOPE_WINDOW)
    if len(data) >= MACD_SLOPE_WINDOW:
        x = np.arange(MACD_SLOPE_WINDOW)
        macd_values = data['MACD'].iloc[-MACD_SLOPE_WINDOW:].values
        signal_values = data['Signal'].iloc[-MACD_SLOPE_WINDOW:].values
        macd_slope = float(np.polyfit(x, macd_values, 1)[0])
        signal_slope = float(np.polyfit(x, signal_values, 1)[0])
        slope_diff_macd = macd_slope - signal_slope

        if macd_values[-1] < 0 and signal_values[-1] < 0:
            selected_val = min(macd_values[-1], signal_values[-1])
        elif macd_values[-1] > 0 and signal_values[-1] > 0:
            selected_val = max(macd_values[-1], signal_values[-1])
        else:
            selected_val = signal_values[-1]
        selected_val = float(selected_val)
    else:
        slope_diff_macd = 0.0
        selected_val = 0.0

    score_MACD = weights['MACD'] * (slope_diff_macd * 10 - selected_val)

    # 3. SMI Signal: using slopes over recent window (with SMI_SLOPE_WINDOW)
    if len(smi) >= SMI_SLOPE_WINDOW:
        x = np.arange(SMI_SLOPE_WINDOW)
        k_values = smi.iloc[-SMI_SLOPE_WINDOW:].values
        d_values = smi_d.iloc[-SMI_SLOPE_WINDOW:].values
        k_slope = float(np.polyfit(x, k_values, 1)[0])
        d_slope = float(np.polyfit(x, d_values, 1)[0])
        slope_diff_smi = k_slope - d_slope
        if k_values[-1] < 0 and d_values[-1] < 0:
            selected_val = min(k_values[-1], d_values[-1])
        elif k_values[-1] > 0 and d_values[-1] > 0:
            selected_val = max(k_values[-1], d_values[-1])
        else:
            selected_val = d_values[-1]
        selected_val = float(selected_val);
    else:
        k_slope = 0.0
        d_slope = 0.0
        slope_diff_smi = 0.0
    score_SMI = weights['SMI'] * (slope_diff_smi - selected_val) / 10

    # 4. Bollinger Bands: deviation of current price from SMA relative to band width
    latest_SMA = float((latest_upper_band + latest_lower_band) / 2) 
    band_width = float(latest_upper_band - latest_lower_band)
    raw_BB = float((latest_SMA - latest_close) / band_width) if band_width != 0 else 0.0
    score_BB = weights['BB'] * raw_BB

    # 5. Ichimoku: ratio difference between current price and mid value
    mid_value = float((latest_senkou_span_a + latest_senkou_span_b) / 2)
    raw_Ichi = float((latest_close - mid_value) / mid_value) if mid_value != 0 else 0.0
    score_Ichi = weights['ICHIMOKU'] * raw_Ichi

    # 6. Divergence: based on RSI and price change
    if latest_close != 0:
        if latest_rsi <= 50:
            raw_div = float((50 - latest_rsi) * ((latest_close - previous_close) / latest_close))
        else:
            raw_div = float(- (latest_rsi - 50) * ((latest_close - previous_close) / latest_close))
    else:
        raw_div = 0
    score_div = weights['DIV'] * raw_div

    # 7. Volume Profile: relative difference between current price and (vp_peak or vp_median)
    selected_vp = min(vp_peak, vp_median) if mode == "conservative" else max(vp_peak, vp_median)
    distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
    raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
    score_VP = weights['VP'] * raw_VP

    # 8. Volume Ratio: based on ratio of today's volume to 20-day average volume
    avg_volume = np.nanmean(data['Volume'].rolling(window=20).mean().values)
    volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1
    raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
    score_Vol = weights['VOL'] * raw_Vol

    # 9. Fibonacci: calculate the relative difference between the latest close and the nearest fibonacci level
    if fib_levels:
        # fib_levels assumed to be a dict with levels as values
        nearest_fib = min(fib_levels.values(), key=lambda level: abs(level - latest_close))
        raw_FIB = (nearest_fib - latest_close) / nearest_fib * 100  # 퍼센트 차이
    else:
        raw_FIB = 0.0
        nearest_fib = None
    score_FIB = weights['FIB'] * raw_FIB

    # Create a list of (raw_value, score) tuples
    indicators_list = [
        (raw_RSI, score_RSI),
        (slope_diff_macd, score_MACD),
        (raw_BB, score_BB),
        (raw_Ichi, score_Ichi),
        (raw_div, score_div),
        (raw_VP, score_VP),
        (raw_Vol, score_Vol),
        (slope_diff_smi, score_SMI),
        (raw_FIB, score_FIB)
    ]

    # Exclude NaN when calculating final score
    final_score = calculate_final_score(indicators_list)

    details = {
        "RSI": {"raw": round(raw_RSI, 2), "score": round(score_RSI, 2)},
        "MACD": {
            "raw": round(slope_diff_macd, 2),
            "score": round(score_MACD, 2),
            "macd": round(latest_macd, 2),
            "signal": round(latest_signal, 2),
            "macd_slope": round(macd_slope, 2) if USE_MACD_SLOPE else 'N/A',
            "signal_slope": round(signal_slope, 2) if USE_MACD_SLOPE else 'N/A'
        },
        "BollingerBands": {"raw": round(raw_BB, 2), "score": round(score_BB, 2)},
        "Ichimoku": {"raw": round(raw_Ichi, 2), "score": round(score_Ichi, 2)},
        "Divergence": {"raw": round(raw_div, 2), "score": round(score_div, 2)},
        "VolumeProfile": {"raw": round(raw_VP, 2), "score": round(score_VP, 2), "vp_peak": round(vp_peak, 2)},
        "VolumeRatio": {"raw": round(raw_Vol, 2), "score": round(score_Vol, 2)},
        "SMI": {
            "raw": round(slope_diff_smi, 2),
            "score": round(score_SMI, 2),
            "K": round(latest_smi, 2),
            "D": round(latest_smi_d, 2),
            "K_slope": round(k_slope, 2),
            "D_slope": round(d_slope, 2)
        },
        "FIB": {"raw": round(raw_FIB, 2), "score": round(score_FIB, 2)},
    }

    print(data["Close"].min(), data["Close"].max())
    min_close = float(data["Close"].min().item())
    max_close = float(data["Close"].max().item())
    scaled_close = ((latest_close - min_close) / (max_close - min_close)) * 200 - 100   

    return {
        "date": latest_date,
        "close": f"{latest_close:.2f}",
        "scaled_close": f"{scaled_close:.2f}",
        "momentum_strength": f"{final_score:.2f}",
        "latest_diff": f"{scaled_close - final_score:.2f}",
        "details": details
    }

def parse_period_to_relativedelta(period_str: str):
    """
    Converts a period string (e.g., '6mo', '1y', '30d') into a relativedelta (or timedelta).
    """
    period_str = period_str.lower().strip()
    if period_str.endswith('y'):
        years = int(period_str[:-1])
        return relativedelta(years=years)
    elif period_str.endswith('mo'):
        months = int(period_str[:-2])
        return relativedelta(months=months)
    elif period_str.endswith('d'):
        days = int(period_str[:-1])
        return timedelta(days=days)
    else:
        raise ValueError("Unsupported period format. Use '1y', '6mo', '30d', etc.")

def analyze_all(data, mode="conservative", analysis_period='1y', normalize=USE_NORMALIZE_MOMENTUM_STRENGTH):
    """
    각 날짜별로, 해당 날짜를 기준으로 analysis_period (예: '6mo') 기간 내의 데이터만 사용하여 momentum strength를 계산합니다.
    """
    weights = get_weight_set(mode)
    # 미리 전체 데이터에 대해 일부 지표를 계산해두는 대신, 각 날짜별로 window_data를 추출하여 계산합니다.
    momentum_strength = {}
    close_prices = {}
    
    period_delta = parse_period_to_relativedelta(analysis_period)
    
    # 전체 데이터는 이미 yfinance에서 가져온 데이터여야 하며, analysis_period*2 이상의 기간을 포함해야 합니다.
    for i in range(1, len(data)):
        try:
            current_date = data.index[i]
            window_start = current_date - period_delta
            window_data = data[(data.index >= window_start) & (data.index <= current_date)]
            if len(window_data) < 2:
                # 충분한 데이터가 없으면 건너뜁니다.
                continue

            # 각 지표별로 window_data에서 재계산
            rsi_series = calculate_rsi(window_data)
            macd_series, signal_series = calculate_macd(window_data)
            upper_band, lower_band = calculate_bollinger_bands(window_data)
            # Ichimoku는 별도 복사본에 계산 (원본 data 변형 방지)
            window_data_copy = window_data.copy()
            calculate_ichimoku(window_data_copy)
            smi_series, smi_d_series = calculate_smi(window_data)
            
            latest_close = float(np.nan_to_num(window_data['Close'].iloc[-1], nan=0.0))
            latest_rsi = float(np.nan_to_num(rsi_series.iloc[-1], nan=50.0))
            latest_macd = float(np.nan_to_num(macd_series.iloc[-1], nan=0.0))
            latest_signal = float(np.nan_to_num(signal_series.iloc[-1], nan=0.0))
            latest_upper_band = float(np.nan_to_num(upper_band.iloc[-1], nan=latest_close))
            latest_lower_band = float(np.nan_to_num(lower_band.iloc[-1], nan=latest_close))
            latest_senkou_span_a = float(np.nan_to_num(window_data_copy['Senkou Span A'].iloc[-1], nan=latest_close))
            latest_senkou_span_b = float(np.nan_to_num(window_data_copy['Senkou Span B'].iloc[-1], nan=latest_close))
            latest_volume = float(np.nan_to_num(window_data['Volume'].iloc[-1], nan=1.0))
            latest_smi = float(np.nan_to_num(smi_series.iloc[-1], nan=0.0))
            latest_smi_d = float(np.nan_to_num(smi_d_series.iloc[-1], nan=0.0))
            previous_close = float(np.nan_to_num(window_data['Close'].iloc[-2], nan=latest_close))
            
            # 1. RSI signal
            raw_RSI = 40 - latest_rsi
            score_RSI = weights['RSI'] * raw_RSI

            # 2. MACD
            if USE_MACD_SLOPE and len(window_data) >= MACD_SLOPE_WINDOW:
                x = np.arange(MACD_SLOPE_WINDOW)
                macd_values = macd_series.iloc[-MACD_SLOPE_WINDOW:].values
                signal_values = signal_series.iloc[-MACD_SLOPE_WINDOW:].values
                macd_slope = float(np.polyfit(x, macd_values, 1)[0])
                signal_slope = float(np.polyfit(x, signal_values, 1)[0])
                slope_diff_macd = macd_slope - signal_slope

                if macd_values[-1] < 0 and signal_values[-1] < 0:
                    selected_val = min(macd_values[-1], signal_values[-1])
                elif macd_values[-1] > 0 and signal_values[-1] > 0:
                    selected_val = max(macd_values[-1], signal_values[-1])
                else:
                    selected_val = signal_values[-1]
                selected_val = float(selected_val)
            else:
                slope_diff_macd = 0.0
                selected_val = 0.0

            score_MACD = weights['MACD'] * (slope_diff_macd * 10 - selected_val)

            # 3. SMI Signal
            if len(smi_series) >= SMI_SLOPE_WINDOW:
                x = np.arange(SMI_SLOPE_WINDOW)
                k_values = smi_series.iloc[-SMI_SLOPE_WINDOW:].values
                d_values = smi_d_series.iloc[-SMI_SLOPE_WINDOW:].values
                k_slope = float(np.polyfit(x, k_values, 1)[0])
                d_slope = float(np.polyfit(x, d_values, 1)[0])
                slope_diff_smi = k_slope - d_slope
                if k_values[-1] < 0 and d_values[-1] < 0:
                    selected_val = min(k_values[-1], d_values[-1])
                elif k_values[-1] > 0 and d_values[-1] > 0:
                    selected_val = max(k_values[-1], d_values[-1])
                else:
                    selected_val = d_values[-1]
                selected_val = float(selected_val);
            else:
                k_slope = 0.0
                d_slope = 0.0
                slope_diff_smi = 0.0
            score_SMI = weights['SMI'] * (slope_diff_smi + selected_val) / 10

            # 4. Bollinger Bands
            latest_SMA = float((latest_upper_band + latest_lower_band) / 2)
            band_width = float(latest_upper_band - latest_lower_band)
            raw_BB = float((latest_SMA - latest_close) / band_width) if band_width != 0 else 0.0
            score_BB = weights['BB'] * raw_BB

            # 5. Ichimoku
            mid_value = float((latest_senkou_span_a + latest_senkou_span_b) / 2)
            raw_Ichi = float((latest_close - mid_value) / mid_value) if mid_value != 0 else 0.0
            score_Ichi = weights['ICHIMOKU'] * raw_Ichi

            # 6. Divergence
            if latest_close != 0:
                if latest_rsi <= 50:
                    raw_div = float((50 - latest_rsi) * ((latest_close - previous_close) / latest_close))
                else:
                    raw_div = float(- (latest_rsi - 50) * ((latest_close - previous_close) / latest_close))
            else:
                raw_div = 0.0
            score_div = weights['DIV'] * raw_div

            # 7. Volume Profile (계산은 window_data 내에서 수행)
            vp_peak, vp_median, _, _ = calculate_volume_profile(window_data, bins=20)
            selected_vp = min(vp_peak, vp_median) if mode == "conservative" else max(vp_peak, vp_median)
            distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
            raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
            score_VP = weights['VP'] * raw_VP

            # 8. Volume Ratio
            rolling_volume = window_data['Volume'].rolling(window=20).mean()
            if rolling_volume.dropna().empty:
                avg_volume = 0.0  # 또는 적절한 기본값 (예: window_data['Volume'].mean())
            else:
                avg_volume = np.nanmean(rolling_volume.values)
            volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1.0
            raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
            score_Vol = weights['VOL'] * raw_Vol

            # 9. Fibonacci: window_data를 사용하여 계산
            fib_levels = calculate_fibonacci_levels(window_data)
            if fib_levels:
                nearest_fib = min(fib_levels.values(), key=lambda level: abs(level - latest_close))
                raw_FIB = (nearest_fib - latest_close) / nearest_fib * 100
            else:
                raw_FIB = 0.0
            score_FIB = weights['FIB'] * raw_FIB

            indicators_list = [
                (raw_RSI, score_RSI),
                (slope_diff_macd, score_MACD),
                (raw_BB, score_BB),
                (raw_Ichi, score_Ichi),
                (raw_div, score_div),
                (raw_VP, score_VP),
                (raw_Vol, score_Vol),
                (slope_diff_smi, score_SMI),
                (raw_FIB, score_FIB)
            ]
            final_score = calculate_final_score(indicators_list)
            date_str = current_date.strftime("%Y-%m-%d")
            momentum_strength[date_str] = final_score
            close_prices[date_str] = round(latest_close, 2)

        except Exception as e:
            print(f"Error processing {current_date}: {e}")

    if normalize:
        momentum_strength = normalize_scores_tanh(momentum_strength)            

    return {
        "close": close_prices,
        "momentum_strength": momentum_strength
    }


@router.get("/{ticker}/momentum", response_model=MomentumResponse)
async def analyze_stock(ticker: str, period: str = '1y', mode: str = "conservative", reference_date=None):
    try:
        data = get_stock_data2(ticker, period, reference_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    analysis_results = analyze(data, mode, reference_date)

    return {
        "ticker": ticker,
        "date": analysis_results['date'],
        "current_price": analysis_results['close'],
        "scaled_price": analysis_results['scaled_close'],
        "momentum_strength": analysis_results['momentum_strength'],
        "latest_diff": analysis_results['latest_diff'],
        "details": analysis_results['details']
    }

def double_period(period: str) -> str:
    """
    주어진 period 문자열을 두 배로 확장합니다.
    예: "1y" -> "2y", "6mo" -> "1y", "3mo" -> "6mo"
    """
    period = period.lower().strip()
    if period.endswith('y'):
        num = int(period[:-1])
        return f"{num * 2}y"
    elif period.endswith('mo'):
        num = int(period[:-2])
        doubled = num * 2
        # 만약 12개월이면 1y로 변환
        if doubled % 12 == 0:
            years = doubled // 12
            return f"{years}y"
        else:
            return f"{doubled}mo"
    elif period.endswith('d'):
        num = int(period[:-1])
        return f"{num * 2}d"
    else:
        raise ValueError("Unsupported period format. Use '1y', '6mo', '30d', etc.")

@router.get("/{ticker}/momentum_all", response_model=MomentumStrengthResponse)
async def analyze_stock_all(ticker: str, period: str = '1y', mode: str = "conservative"):
    try:
        # 분석에 필요한 올바른 과거 데이터를 위해 period의 두 배 기간의 데이터를 다운로드합니다.
        extended_period = double_period(period)
        data = get_stock_data2(ticker, period=extended_period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    # 분석 시각마다 analysis_period(예: '6mo') 동안의 데이터만 사용하여 계산합니다.
    history = analyze_all(data, mode, analysis_period=period)

    return {"ticker": ticker, "history": history}
