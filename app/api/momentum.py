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
            "ICHIMOKU": 20,
            "DIV": 2,
            "VP": 2,
            "VOL": 2,
            "OBV": 2,
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

""" def calculate_macd_score(macd_series, signal_series, bonus_scale=1.0):
    #macd_series, signal_series: pandas Series (최근 데이터를 포함)
    #MACD_SLOPE_WINDOW: 기울기 계산에 사용할 데이터 포인트 수
    #bonus_scale: 전환 구간 보너스의 배율 조절 (필요에 따라 조정)

    # 최근 데이터 사용
    x = np.arange(MACD_SLOPE_WINDOW)
    macd_values = macd_series.iloc[-MACD_SLOPE_WINDOW:].values
    signal_values = signal_series.iloc[-MACD_SLOPE_WINDOW:].values

    # 기본 점수: 마지막 값에서 (macd - signal)
    base_score = macd_values[-1] - signal_values[-1]

    # 상수 설정 (예시로 계산된 값)
    A = 2.83   # 약 exp(1.04)
    b = 0.52   # 약 ln(8)/4

    # 업/다운 구간에 따른 multiplier 적용
    if macd_values[-1] > signal_values[-1]:
        # 상승 구간: macd가 낮을수록 multiplier가 커짐
        multiplier = A * np.exp(-b * macd_values[-1])
    else:
        # 하락 구간: macd가 높을수록 multiplier가 커짐
        multiplier = A * np.exp(b * macd_values[-1])
    
    adjusted_base_score = base_score * multiplier

    bonus = 0.0

    # 하락추세( macd < signal)에서 전환 보너스 계산 (기존 로직)
    if macd_values[-1] < signal_values[-1]:
        if MACD_SLOPE_WINDOW >= 3:
            x_prev = np.arange(MACD_SLOPE_WINDOW - 1)
            prev_macd_slope = float(np.polyfit(x_prev, macd_values[:-1], 1)[0])
        else:
            prev_macd_slope = 0.0

        current_macd_slope = float(np.polyfit(x, macd_values, 1)[0])
        current_signal_slope = float(np.polyfit(x, signal_values, 1)[0])

        # TODO: 전환조건 오류!!
        # 전환 조건: 이전 macd slope가 음수이고 현재가 양수
        if prev_macd_slope < 0 and current_macd_slope > 0:
            # macd가 signal을 상향 돌파하는 지점 탐색
            crossing_index = None
            for i in range(1, len(macd_values)):
                if macd_values[i-1] < signal_values[i-1] and macd_values[i] >= signal_values[i]:
                    crossing_index = i
                    break

            print('전환발생1: ', crossing_index, current_signal_slope)
            # 돌파가 발생하고 signal slope가 양수이면 보너스 부여 (양수)
            if crossing_index is not None and current_signal_slope > 0:
                macd_at_crossing = macd_values[crossing_index]
                if macd_at_crossing < 0:
                    bonus = bonus_scale * (np.exp(-macd_at_crossing) - 1)
                    print('전환보너스1-1: ', bonus)
                else:
                    bonus = bonus_scale * (np.exp(macd_at_crossing) - 1)
                    print('전환보너스1-2: ', bonus)

    # 상승추세( macd > signal)에서 전환 보너스 계산 (하락 전환, 반대 로직)
    elif macd_values[-1] > signal_values[-1]:
        if MACD_SLOPE_WINDOW >= 3:
            x_prev = np.arange(MACD_SLOPE_WINDOW - 1)
            prev_macd_slope = float(np.polyfit(x_prev, macd_values[:-1], 1)[0])
        else:
            prev_macd_slope = 0.0

        current_macd_slope = float(np.polyfit(x, macd_values, 1)[0])
        current_signal_slope = float(np.polyfit(x, signal_values, 1)[0])

        # 전환 조건: 이전 macd slope가 양수이고 현재가 음수 (즉, 하락 전환)
        if prev_macd_slope > 0 and current_macd_slope < 0:
            # macd가 signal을 하향 돌파하는 지점 탐색
            crossing_index = None
            for i in range(1, len(macd_values)):
                if macd_values[i-1] > signal_values[i-1] and macd_values[i] <= signal_values[i]:
                    crossing_index = i
                    break

            print('전환발생2: ', crossing_index, current_signal_slope)
            # 돌파가 발생하고 signal slope가 음수이면 보너스 부여 (음수로 적용)
            if crossing_index is not None and current_signal_slope < 0:
                macd_at_crossing = macd_values[crossing_index]
                if macd_at_crossing > 0:
                    bonus = - bonus_scale * (np.exp(macd_at_crossing) - 1)
                    print('전환보너스2-1: ', bonus)
                else:
                    bonus = - bonus_scale * (np.exp(-macd_at_crossing) - 1)
                    print('전환보너스2-2: ', bonus)

    # 최종 점수 = 보정된 기본 점수 + 보너스 (상황에 따라 보너스는 양수 또는 음수)
    final_score = adjusted_base_score + bonus
    final_score = max(min(final_score, 20), -20)
    
    return final_score """

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
    """
    Analyzes the stock data for the given reference date.
    Only uses the data up to and including the reference date.
    """
    tz = timezone('Canada/Mountain')
    weights = get_weight_set(mode)
    
    # 기준일 설정: 없으면 현재 시간, 문자열 등은 datetime으로 변환
    if reference_date is None:
        reference_date = datetime.now(tz)
    elif not isinstance(reference_date, datetime):
        reference_date = pd.to_datetime(reference_date).to_pydatetime()
        reference_date = reference_date.astimezone(tz)
    
    # pandas Timestamp로 변환 후 tz 정보를 제거하여 tz-naive로 만듦
    reference_date = pd.to_datetime(reference_date).tz_localize(None)
    
    # data.index도 pandas Timestamp로 변환 후 tz 정보 제거
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    # reference_date보다 작거나 같은 모든 날짜를 추출 후 가장 최근 날짜 선택
    valid_dates = data.index[data.index <= reference_date]
    if valid_dates.empty:
        closest_date = data.index[0]
    else:
        closest_date = valid_dates.max()
    
    # closest_date까지의 데이터를 선택
    window_data = data.loc[:closest_date]
    
    # If there is not enough data in the window, return early
    if len(window_data) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for analysis.")
    
    # Calculate indicators for the window_data
    rsi_series = calculate_rsi(window_data)
    macd_series, signal_series = calculate_macd(window_data)
    upper_band, lower_band = calculate_bollinger_bands(window_data)
    window_data_copy = window_data.copy()  # To avoid modifying original data
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
    
    # Perform calculations for each indicator using the window data
    raw_RSI = 40 - latest_rsi
    score_RSI = weights['RSI'] * raw_RSI

    #macd_score = calculate_macd_score(macd_series, signal_series)
    #print("MACD score: ", macd_score)

    if USE_MACD_SLOPE and len(window_data) >= MACD_SLOPE_WINDOW:
        x = np.arange(MACD_SLOPE_WINDOW)
        macd_values = macd_series.iloc[-MACD_SLOPE_WINDOW:].values
        signal_values = signal_series.iloc[-MACD_SLOPE_WINDOW:].values
        macd_slope = float(np.polyfit(x, macd_values, 1)[0])
        signal_slope = float(np.polyfit(x, signal_values, 1)[0])

        macd_last = float(macd_values[-1])
        signal_last = float(signal_values[-1])

        value_diff_macd = (macd_last - signal_last) * 20
        slope_diff_macd = macd_slope - signal_slope

        if macd_last > signal_last:
            if macd_slope > signal_slope:
                raw_MACD = (value_diff_macd + slope_diff_macd) / 3
            else:
                raw_MACD = slope_diff_macd / 2
        else:
            if macd_slope < signal_slope:
                raw_MACD = (value_diff_macd + slope_diff_macd) / 3
            else:
                raw_MACD = slope_diff_macd / 2
    else:
        macd_slope = 0.0
        signal_slope = 0.0
        raw_MACD = 0.0

    score_MACD = weights['MACD'] * raw_MACD
    score_MACD = max(min(score_MACD, 20), -20)

    if len(smi_series) >= SMI_SLOPE_WINDOW:
        x = np.arange(SMI_SLOPE_WINDOW)
        k_values = smi_series.iloc[-SMI_SLOPE_WINDOW:].values
        d_values = smi_d_series.iloc[-SMI_SLOPE_WINDOW:].values
        k_slope = float(np.polyfit(x, k_values, 1)[0])
        d_slope = float(np.polyfit(x, d_values, 1)[0])
        
        k_last = float(k_values[-1])
        d_last = float(d_values[-1])
        
        value_diff_smi = k_last - d_last
        slope_diff_smi = k_slope - d_slope
        
        if k_last > d_last:
            if k_slope > d_slope:
                raw_SMI = (value_diff_smi + slope_diff_smi) / 3
            else:
                raw_SMI = slope_diff_smi / 2
        else:
            if k_slope < d_slope:
                raw_SMI = (value_diff_smi + slope_diff_smi) / 3
            else:
                raw_SMI = slope_diff_smi / 2
    else:
        k_slope = 0.0
        d_slope = 0.0
        raw_SMI = 0.0

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

    vp_peak, vp_median, _, _ = calculate_volume_profile(window_data, bins=20)
    selected_vp = min(vp_peak, vp_median) if mode == "conservative" else max(vp_peak, vp_median)
    distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
    raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
    score_VP = weights['VP'] * raw_VP
    score_VP = max(min(score_VP, 20), -20)

    rolling_volume = window_data['Volume'].rolling(window=20).mean()
    avg_volume = np.nanmean(rolling_volume.values) if not rolling_volume.dropna().empty else 0.0
    volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1.0
    raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
    score_Vol = weights['VOL'] * raw_Vol

    obv_series = compute_obv(window_data)
    obv_latest = float(obv_series.iloc[-1])
    obv_min = float(obv_series.min())
    obv_max = float(obv_series.max())
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
        (slope_diff_macd, score_MACD),
        (raw_BB, score_BB),
        (raw_Ichi, score_Ichi),
        (raw_div, score_div),
        (raw_VP, score_VP),
        (raw_Vol, score_Vol),
        (slope_diff_smi, score_SMI),
        (raw_FIB, score_FIB),
        (norm_OBV, score_OBV)
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
        "VolumeProfile": {"raw": round(raw_VP, 2), "score": round(score_VP, 2), "vp_peak": round(vp_peak, 2), "vp_median": round(vp_median, 2)},
        "BalanceVolumeRatio": {"raw": round(raw_Vol, 2), "score": round(score_Vol, 2)},
        "OBV": {"raw": round(norm_OBV, 2), "score": round(score_OBV, 2)},
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

                macd_last = float(macd_values[-1])
                signal_last = float(signal_values[-1])

                value_diff_macd = (macd_last - signal_last) * 20
                slope_diff_macd = macd_slope - signal_slope

                if macd_last > signal_last:
                    if macd_slope > signal_slope:
                        raw_MACD = (value_diff_macd + slope_diff_macd) / 3
                    else:
                        raw_MACD = slope_diff_macd / 2
                else:
                    if macd_slope < signal_slope:
                        raw_MACD = (value_diff_macd + slope_diff_macd) / 3
                    else:
                        raw_MACD = slope_diff_macd / 2
            else:
                macd_slope = 0.0
                signal_slope = 0.0
                raw_MACD = 0.0

            score_MACD = weights['MACD'] * raw_MACD
            score_MACD = max(min(score_MACD, 20), -20)

            if len(smi_series) >= SMI_SLOPE_WINDOW:
                x = np.arange(SMI_SLOPE_WINDOW)
                k_values = smi_series.iloc[-SMI_SLOPE_WINDOW:].values
                d_values = smi_d_series.iloc[-SMI_SLOPE_WINDOW:].values
                k_slope = float(np.polyfit(x, k_values, 1)[0])
                d_slope = float(np.polyfit(x, d_values, 1)[0])
                
                k_last = float(k_values[-1])
                d_last = float(d_values[-1])
                
                value_diff_smi = k_last - d_last
                slope_diff_smi = k_slope - d_slope
                
                if k_last > d_last:
                    if k_slope > d_slope:
                        raw_SMI = (value_diff_smi + slope_diff_smi) / 3
                    else:
                        raw_SMI = slope_diff_smi / 2
                else:
                    if k_slope < d_slope:
                        raw_SMI = (value_diff_smi + slope_diff_smi) / 3
                    else:
                        raw_SMI = slope_diff_smi / 2
            else:
                k_slope = 0.0
                d_slope = 0.0
                raw_SMI = 0.0

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

            # 7. Volume Profile (계산은 window_data 내에서 수행)
            vp_peak, vp_median, _, _ = calculate_volume_profile(window_data, bins=20)
            selected_vp = min(vp_peak, vp_median) if mode == "conservative" else max(vp_peak, vp_median)
            distance = (latest_close - selected_vp) / selected_vp if selected_vp != 0 else 0.0
            raw_VP = - distance * 10 if distance > 0 else abs(distance) * 10
            score_VP = weights['VP'] * raw_VP
            score_VP = max(min(score_VP, 20), -20)

            # 8. Volume Ratio
            rolling_volume = window_data['Volume'].rolling(window=20).mean()
            if rolling_volume.dropna().empty:
                avg_volume = 0.0  # 또는 적절한 기본값 (예: window_data['Volume'].mean())
            else:
                avg_volume = np.nanmean(rolling_volume.values)
            volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1.0
            raw_Vol = - (volume_ratio - 1) if latest_close < previous_close else (volume_ratio - 1)
            score_Vol = weights['VOL'] * raw_Vol

            # 9. On-Balance Volume
            obv_series = compute_obv(window_data)
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
                (slope_diff_macd, score_MACD),
                (raw_BB, score_BB),
                (raw_Ichi, score_Ichi),
                (raw_div, score_div),
                (raw_VP, score_VP),
                (raw_Vol, score_Vol),
                (slope_diff_smi, score_SMI),
                (raw_FIB, score_FIB),
                (norm_OBV, score_OBV)
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
