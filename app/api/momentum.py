import yfinance as yf
import pandas as pd
import numpy as np
import math
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
            "MACD": 4,
            "SMI": 2,
            "BB": 10,
            "ICHIMOKU": 20,
            "DIV": 1,
            "VP": 1,
            "VOL": 2,
            "OBV": 2,
            "FIB": 0,   # exclude FIB 4,
        }
    else:  # Default: conservative
        return {
            "RSI": 1,
            "MACD":  1,
            "SMI": 1,
            "BB": 10,
            "ICHIMOKU": 40,
            "DIV": 2,
            "VP": 2,
            "VOL": 0,   # exclude Volume Ratio 2,
            "OBV": 1,
            "FIB": 0,   # exclude FIB 1,
        }

# Default settings
USE_NORMALIZE_MOMENTUM_STRENGTH = False

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
    # Return NaN if there isn't enough data (requires window*2 days)
    if len(data) < window * 2:
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

    # Compute histogram for volume profile
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    max_index = np.argmax(hist)
    vp_peak = (bin_edges[max_index] + bin_edges[max_index+1]) / 2

    # Calculate weighted median from raw data
    vp_median = weighted_median(prices, volumes)

    # Calculate volume-weighted average price
    vp_average = np.average(prices, weights=volumes)
    return vp_peak, vp_median, vp_average, hist, bin_edges

def calculate_smi(data, k_period=10, k_smoothing=3, k_double_smoothing=3, d_period=10):
    highest_high = data['High'].rolling(window=k_period, min_periods=1).max()
    lowest_low = data['Low'].rolling(window=k_period, min_periods=1).min()
    
    median = (highest_high + lowest_low) / 2
    
    raw = data['Close'] - median
    raw_range = (highest_high - lowest_low) / 2
    
    smooth_raw = raw.ewm(span=k_smoothing, adjust=False).mean()
    smooth_range = raw_range.ewm(span=k_smoothing, adjust=False).mean()
    
    double_smooth_raw = smooth_raw.ewm(span=k_double_smoothing, adjust=False).mean()
    double_smooth_range = smooth_range.ewm(span=k_double_smoothing, adjust=False).mean()
    
    double_smooth_range = double_smooth_range.replace(0, np.nan)
    
    smi = 100 * (double_smooth_raw / double_smooth_range)
    
    smi_d = smi.ewm(span=d_period, adjust=False).mean()
    
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


def compute_obv(data):
    obv_values = [0]
    close_series = data["Close"].squeeze()
    volume_series = data["Volume"].squeeze()
    for i in range(1, len(close_series)):
        if close_series.iat[i] > close_series.iat[i-1]:
            obv_values.append(obv_values[-1] + volume_series.iat[i])
        elif close_series.iat[i] < close_series.iat[i-1]:
            obv_values.append(obv_values[-1] - volume_series.iat[i])
        else:
            obv_values.append(obv_values[-1])
    return pd.Series(obv_values, index=data.index)

def compute_moving_obv(data, window=120):
    # Select the last 'window' days and reset the index to ensure consecutive integer index
    window_data = data.iloc[-window:].reset_index(drop=True)
    close_series = window_data["Close"]
    volume_series = window_data["Volume"]
    
    # Ensure these are Series; if they are DataFrames, convert them to Series by taking the first column.
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    if isinstance(volume_series, pd.DataFrame):
        volume_series = volume_series.iloc[:, 0]
    
    obv_values = [0]
    for i in range(1, len(close_series)):
        if close_series.iat[i] > close_series.iat[i-1]:
            obv_values.append(obv_values[-1] + volume_series.iat[i])
        elif close_series.iat[i] < close_series.iat[i-1]:
            obv_values.append(obv_values[-1] - volume_series.iat[i])
        else:
            obv_values.append(obv_values[-1])
            
    # Return with the same (reset) index
    return pd.Series(obv_values, index=window_data.index)

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

def exponential_scale(score, min_val, max_val, alpha=1.0, range=20):
    normalized = (score - min_val) / (max_val - min_val)
    normalized = 2 * normalized - 1
    scaled = range * math.copysign((math.exp(alpha * abs(normalized)) - 1) / (math.exp(alpha) - 1), normalized)
    return scaled

def calculate_final_score(indicators):
    # Add only non-NaN scores to valid_scores
    valid_scores = [score for raw, score in indicators if not np.isnan(raw)]
    # TODO: 모멘텀 가중치 적용 ((smi + macd) * obv) / x
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

def dual_signal_score(
    series_a: pd.Series,
    series_b: pd.Series,
    window_slope: int = 5,
    weight_primary: float = -0.8,     # Weight for the primary indicator (x)
    weight_slope: float = 3.0,        # Weight for the slope of the primary indicator (y)
    weight_diff: float = 3.0,         # Weight for the difference between the primary and secondary indicators (z)
    weight_synergy: float = 10.0,     # Weight for the synergy term
    weight_diff_change: float = 8.0,  # Weight for the adjustment based on the change in difference
    scaling_factor: float = 10.0      # Scaling factor to avoid tanh saturation
) -> pd.Series:
    """
    A formula-based scoring function for two related signals that calculates:
      Score = 10 * tanh( S_eff )
    
    where S_eff = (weight_primary*x + weight_slope*y + weight_diff*z + weight_synergy*synergy + diff_adjustment) / scaling_factor,
    
    and:
      x = primary indicator value (from series_a)
      y = slope of the primary indicator over 'window_slope' bars (computed from series_a)
      z = difference between the primary and secondary indicators (series_a - series_b)
      synergy = max(0, -x) * max(0, y) * max(0, z)
      diff_adjustment = weight_diff_change * (diff[i] - diff[i-1])
    
    The diff_adjustment reflects that:
      - In an upward process (i.e. when the primary indicator > secondary indicator),
        if the difference increases then the score is raised, and if it decreases then the score is lowered.
      - Similarly, in a downward process, if the gap narrows the score increases,
        and if the gap widens the score decreases.
    
    Parameters:
      series_a: Series of primary indicator values.
      series_b: Series of secondary indicator values.
      window_slope: Number of bars used to compute the slope of the primary indicator.
      weight_primary, weight_slope, weight_diff, weight_synergy: Coefficients for the respective terms.
      weight_diff_change: Coefficient for the adjustment based on the change in difference.
      scaling_factor: Factor to scale the combined S to avoid tanh saturation.
    
    Returns:
      A pd.Series of scores in the range [-10, 10].
    """
    # Align the lengths of the two series
    length = min(len(series_a), len(series_b))
    series_a = series_a.iloc[-length:]
    series_b = series_b.iloc[-length:]
    
    primary_values = series_a.values
    secondary_values = series_b.values

    # Check if enough data exists for slope calculation
    if length < window_slope + 1:
        return pd.Series([np.nan] * length, index=series_a.index)
    
    # Compute the slope of the primary indicator using a simple difference method over window_slope bars
    slopes = np.full(length, np.nan, dtype=float)
    for i in range(window_slope, length):
        slopes[i] = (primary_values[i] - primary_values[i - window_slope]) / window_slope

    scores = np.zeros(length, dtype=float)
    # Calculate the difference between the primary and secondary indicators for each point
    diff_values = primary_values - secondary_values

    for i in range(length):
        x = primary_values[i]
        y = slopes[i] if not np.isnan(slopes[i]) else 0.0
        z = x - secondary_values[i]
        
        # Synergy term: active when the primary indicator is negative,
        # its slope is positive, and it is above the secondary indicator.
        synergy_val = max(0.0, -x) * max(0.0, y) * max(0.0, z)
        
        # Compute the adjustment based on the change in difference (if available)
        if i > 0:
            diff_change = diff_values[i] - diff_values[i - 1]
            diff_adjustment = weight_diff_change * diff_change
        else:
            diff_adjustment = 0.0

        # Combine all terms linearly
        S = (weight_primary * x +
             weight_slope * y +
             weight_diff * z +
             weight_synergy * synergy_val +
             diff_adjustment)

        # Scale S to avoid tanh saturation
        S_eff = S / scaling_factor

        # Final score is bounded between -10 and +10 via tanh
        score = 10.0 * np.tanh(S_eff)
        scores[i] = score

    return pd.Series(scores, index=series_a.index)

def analyze(data, mode="conservative", reference_date=None):
    """
    Analyzes the stock data for the given reference date.
    Only uses the data up to and including the reference date.
    """
    tz = timezone('Canada/Mountain')
    weights = get_weight_set(mode)
    
    if reference_date is None:
        reference_date = datetime.now(tz)
    elif not isinstance(reference_date, datetime):
        reference_date = pd.to_datetime(reference_date).to_pydatetime()
        reference_date = reference_date.astimezone(tz)
    
    reference_date = pd.to_datetime(reference_date).tz_localize(None)
    
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    valid_dates = data.index[data.index <= reference_date]
    if valid_dates.empty:
        closest_date = data.index[0]
    else:
        closest_date = valid_dates.max()
    
    window_data = data.loc[:closest_date]
    
    # If there is not enough data in the window, return early
    if len(window_data) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for analysis.")
    
    # Calculate indicators for the window_data
    rsi_series = calculate_rsi(window_data)
    upper_band, lower_band = calculate_bollinger_bands(window_data)
    window_data_copy = window_data.copy()  # Copy data to preserve the original for Ichimoku calculations.
    calculate_ichimoku(window_data_copy)
    macd_series, signal_series = calculate_macd(window_data)
    smi_k_series, smi_d_series = calculate_smi(window_data)
    
    latest_close = float(np.nan_to_num(window_data['Close'].iloc[-1], nan=0.0))
    latest_rsi = float(np.nan_to_num(rsi_series.iloc[-1], nan=50.0))
    max_rsi = float(np.nan_to_num(rsi_series.max(), nan=50.0))
    min_rsi = float(np.nan_to_num(rsi_series.min(), nan=50.0))
    latest_upper_band = float(np.nan_to_num(upper_band.iloc[-1], nan=latest_close))
    latest_lower_band = float(np.nan_to_num(lower_band.iloc[-1], nan=latest_close))
    latest_senkou_span_a = float(np.nan_to_num(window_data_copy['Senkou Span A'].iloc[-1], nan=latest_close))
    latest_senkou_span_b = float(np.nan_to_num(window_data_copy['Senkou Span B'].iloc[-1], nan=latest_close))
    latest_volume = float(np.nan_to_num(window_data['Volume'].iloc[-1], nan=1.0))
    latest_macd = float(np.nan_to_num(macd_series.iloc[-1], nan=0.0))
    latest_signal = float(np.nan_to_num(signal_series.iloc[-1], nan=0.0))
    latest_smi_k = float(np.nan_to_num(smi_k_series.iloc[-1], nan=0.0))
    latest_smi_d = float(np.nan_to_num(smi_d_series.iloc[-1], nan=0.0))
    previous_close = float(np.nan_to_num(window_data['Close'].iloc[-2], nan=latest_close))

    raw_RSI = -latest_rsi
    score_RSI = exponential_scale(raw_RSI, -max_rsi, -min_rsi)

    macd_scores = dual_signal_score(macd_series, signal_series)
    raw_MACD = macd_scores.iloc[-1]
    score_MACD = weights['MACD'] * raw_MACD

    smi_scores = dual_signal_score(smi_k_series, smi_d_series)
    raw_SMI = smi_scores.iloc[-1]
    score_SMI = weights['SMI'] * raw_SMI

    latest_SMA = float((latest_upper_band + latest_lower_band) / 2)
    band_width = float(latest_upper_band - latest_lower_band)
    raw_BB = float((latest_SMA - latest_close) / band_width) if band_width != 0 else 0.0
    score_BB = weights['BB'] * raw_BB

    mid_value = float((latest_senkou_span_a + latest_senkou_span_b) / 2)
    raw_Ichi = float((latest_close - mid_value) / mid_value) if mid_value != 0 else 0.0
    score_Ichi = weights['ICHIMOKU'] * raw_Ichi

    if latest_close != 0:
        if latest_rsi <= 50:
            raw_div = float((50 - latest_rsi) * ((latest_close - previous_close) / latest_close))
        else:
            raw_div = float(- (latest_rsi - 50) * ((latest_close - previous_close) / latest_close))
    else:
        raw_div = 0.0
    score_div = weights['DIV'] * raw_div

    vp_peak, vp_median, vp_average, _, _ = calculate_volume_profile(window_data, bins=20)
    selected_vp = min(vp_peak, vp_average) if mode == "conservative" else max(vp_peak, vp_median)
    distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
    raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
    score_VP = weights['VP'] * raw_VP
    score_VP = max(min(score_VP, 20), -20)

    rolling_volume = window_data['Volume'].rolling(window=20).mean()
    avg_volume = np.nanmean(rolling_volume.values) if not rolling_volume.dropna().empty else 0.0
    volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1.0
    raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
    score_Vol = weights['VOL'] * raw_Vol

    #obv_series = compute_obv(window_data)
    obv_series = compute_moving_obv(window_data)    
    obv_latest = float(obv_series.iloc[-1])
    obv_min = float(obv_series.min())
    obv_max = float(obv_series.max())
    print('OBV: ', obv_series[-10:], 'latest: ', obv_latest, 'min: ', obv_min, 'max: ', obv_max, 'raw: ', ((obv_latest - obv_min) / (obv_max - obv_min)), 'score: ', ((obv_latest - obv_min) / (obv_max - obv_min)) * 20 - 10)
    if obv_max != obv_min:
        norm_OBV = ((obv_latest - obv_min) / (obv_max - obv_min)) * 20 - 10
    else:
        norm_OBV = 0.0
    score_OBV = weights['OBV'] * norm_OBV

    fib_levels = calculate_fibonacci_levels(window_data)
    if fib_levels:
        nearest_fib = min(fib_levels.values(), key=lambda level: abs(level - latest_close))
        raw_FIB = (nearest_fib - latest_close) / nearest_fib * 100
    else:
        raw_FIB = 0.0
    score_FIB = weights['FIB'] * raw_FIB

    indicators_list = [
        (raw_RSI, score_RSI),
        (raw_BB, score_BB),
        (raw_Ichi, score_Ichi),
        (raw_div, score_div),
        (raw_VP, score_VP),
        (raw_Vol, score_Vol),
        (norm_OBV, score_OBV),
        (raw_MACD, score_MACD),
        (raw_SMI, score_SMI),
        (raw_FIB, score_FIB)
    ]

    # Exclude NaN when calculating final score
    final_score = calculate_final_score(indicators_list)

    details = {
        "RSI": {"raw": round(raw_RSI, 2), "score": round(score_RSI, 2), "latest": round(latest_rsi, 2), "min": round(min_rsi, 2), "max": round(max_rsi, 2)},
        "BollingerBands": {"raw": round(raw_BB, 2), "score": round(score_BB, 2)},
        "Ichimoku": {"raw": round(raw_Ichi, 2), "score": round(score_Ichi, 2)},
        "Divergence": {"raw": round(raw_div, 2), "score": round(score_div, 2)},
        "VolumeProfile": {"raw": round(raw_VP, 2), "score": round(score_VP, 2), "vp_peak": round(vp_peak, 2), "vp_median": round(vp_median, 2), "vp_average": round(vp_average, 2)},
        "BalanceVolumeRatio": {"raw": round(raw_Vol, 2), "score": round(score_Vol, 2)},
        "OBV": {"raw": round(norm_OBV, 2), "score": round(score_OBV, 2)},
        "MACD": {"raw": round(raw_MACD, 2), "score": round(score_MACD, 2)},
        "SMI": {"raw": round(raw_SMI, 2), "score": round(score_SMI, 2)},
        "FIB": {"raw": round(raw_FIB, 2), "score": round(score_FIB, 2)},
    }

    print(data["Close"].min(), data["Close"].max())
    min_close = float(data["Close"].min().item())
    max_close = float(data["Close"].max().item())
    scaled_close = ((latest_close - min_close) / (max_close - min_close)) * 200 - 100   

    return {
        "date": reference_date.strftime("%Y-%m-%d"),
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

def analyze_all(data, mode="conservative", analysis_period='6mo', normalize=USE_NORMALIZE_MOMENTUM_STRENGTH):
    weights = get_weight_set(mode)
    momentum_strength = {}
    close_prices = {}
    
    period_delta = parse_period_to_relativedelta(analysis_period)
    offset = math.ceil(len(data) / 2) # data has double period

    for i in range(1, len(data) // 2): # data has double period
        try:
            current_date = data.index[i + offset]
            window_start = current_date - period_delta
            window_data = data[(data.index >= window_start) & (data.index <= current_date)]
            if len(window_data) < 2:
                continue

            rsi_series = calculate_rsi(window_data)
            upper_band, lower_band = calculate_bollinger_bands(window_data)
            window_data_copy = window_data.copy()
            calculate_ichimoku(window_data_copy)
            macd_series, signal_series = calculate_macd(window_data)
            smi_k_series, smi_d_series = calculate_smi(window_data)
            
            latest_close = float(np.nan_to_num(window_data['Close'].iloc[-1], nan=0.0))
            latest_rsi = float(np.nan_to_num(rsi_series.iloc[-1], nan=50.0))
            max_rsi = float(np.nan_to_num(rsi_series.max(), nan=50.0))
            min_rsi = float(np.nan_to_num(rsi_series.min(), nan=50.0))
            latest_upper_band = float(np.nan_to_num(upper_band.iloc[-1], nan=latest_close))
            latest_lower_band = float(np.nan_to_num(lower_band.iloc[-1], nan=latest_close))
            latest_senkou_span_a = float(np.nan_to_num(window_data_copy['Senkou Span A'].iloc[-1], nan=latest_close))
            latest_senkou_span_b = float(np.nan_to_num(window_data_copy['Senkou Span B'].iloc[-1], nan=latest_close))
            latest_volume = float(np.nan_to_num(window_data['Volume'].iloc[-1], nan=1.0))
            latest_macd = float(np.nan_to_num(macd_series.iloc[-1], nan=0.0))
            latest_signal = float(np.nan_to_num(signal_series.iloc[-1], nan=0.0))
            latest_smi = float(np.nan_to_num(smi_k_series.iloc[-1], nan=0.0))
            latest_smi_d = float(np.nan_to_num(smi_d_series.iloc[-1], nan=0.0))
            previous_close = float(np.nan_to_num(window_data['Close'].iloc[-2], nan=latest_close))
            
            # 1. RSI signal
            raw_RSI = -latest_rsi
            score_RSI = exponential_scale(raw_RSI, -max_rsi, -min_rsi)

            # 2. MACD
            macd_scores = dual_signal_score(macd_series, signal_series)
            raw_MACD = macd_scores.iloc[-1]
            score_MACD = weights['MACD'] * raw_MACD

            # 3. SMI
            smi_scores = dual_signal_score(smi_k_series, smi_d_series)
            raw_SMI = smi_scores.iloc[-1]
            score_SMI = weights['SMI'] * raw_SMI

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

            # 7. Volume Profile
            vp_peak, vp_median, vp_average, _, _ = calculate_volume_profile(window_data, bins=20)
            selected_vp = min(vp_peak, vp_average) if mode == "conservative" else max(vp_peak, vp_median)
            distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
            raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
            score_VP = weights['VP'] * raw_VP
            score_VP = max(min(score_VP, 20), -20)

            # 8. Volume Ratio
            rolling_volume = window_data['Volume'].rolling(window=20).mean()
            if rolling_volume.dropna().empty:
                avg_volume = 0.0 # window_data['Volume'].mean()?? instead of 0.0
            else:
                avg_volume = np.nanmean(rolling_volume.values)
            volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1.0
            raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
            score_Vol = weights['VOL'] * raw_Vol

            # 9. On-Balance Volume
            #obv_series = compute_obv(window_data)
            obv_series = compute_moving_obv(window_data)
            obv_latest = float(obv_series.iloc[-1])
            obv_min = float(obv_series.min())
            obv_max = float(obv_series.max())
            if obv_max != obv_min:
                norm_OBV = ((obv_latest - obv_min) / (obv_max - obv_min)) * 20 - 10
            else:
                norm_OBV = 0.0
            score_OBV = weights['OBV'] * norm_OBV

            # 10. Fibonacci Level
            fib_levels = calculate_fibonacci_levels(window_data)
            if fib_levels:
                nearest_fib = min(fib_levels.values(), key=lambda level: abs(level - latest_close))
                raw_FIB = (nearest_fib - latest_close) / nearest_fib * 100
            else:
                raw_FIB = 0.0
            score_FIB = weights['FIB'] * raw_FIB

            indicators_list = [
                (raw_RSI, score_RSI),
                (raw_BB, score_BB),
                (raw_Ichi, score_Ichi),
                (raw_div, score_div),
                (raw_VP, score_VP),
                (raw_Vol, score_Vol),
                (norm_OBV, score_OBV),
                (raw_MACD, score_MACD),
                (raw_SMI, score_SMI),
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
    period = period.lower().strip()
    if period.endswith('y'):
        num = int(period[:-1])
        return f"{num * 2}y"
    elif period.endswith('mo'):
        num = int(period[:-2])
        doubled = num * 2

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
async def analyze_stock_all(ticker: str, period: str = '1y', mode: str = "conservative", reference_date=None):
    try:
        # Download data for twice the period to obtain proper historical data for analysis.
        extended_period = double_period(period)
        data = get_stock_data2(ticker, extended_period, reference_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    # For each analysis point, only use data from the analysis period (e.g., '6mo') for calculations.
    history = analyze_all(data, mode, analysis_period=period)

    return {"ticker": ticker, "history": history}
