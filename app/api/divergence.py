import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Individual weights for each indicator (adjust as needed)
WEIGHT_RSI = 1 #2
WEIGHT_MACD = 10
WEIGHT_BB = 10
WEIGHT_ICHIMOKU = 20
WEIGHT_DIV = 1
WEIGHT_VP = 1 #10
WEIGHT_VOL = 2 #10
WEIGHT_SMI = 2 #10

# Configurable slope window sizes
MACD_SLOPE_WINDOW = 5
SMI_SLOPE_WINDOW = 5

class TrendScoreResponse(BaseModel):
    ticker: str
    trend_scores: dict  # Date -> trend score

class DivergenceResponse(BaseModel):
    ticker: str
    date: str
    current_price: str
    trend_score: str  # Final score in range -100 to 100 (string, 2 decimals)
    details: dict

def calculate_rsi(data, window=14):
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
    difference = max_price - min_price
    level1 = max_price - difference * 0.236
    level2 = max_price - difference * 0.382
    level3 = max_price - difference * 0.618
    return level1, level2, level3

def calculate_volume_profile(data, bins=20):
    prices = data['Close'].values
    volumes = data['Volume'].values
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    max_index = np.argmax(hist)
    vp_peak = (bin_edges[max_index] + bin_edges[max_index+1]) / 2
    return vp_peak, hist, bin_edges

def calculate_smi(data, k_period=5, d_period=3, smoothing=3):
    M = (data['High'] + data['Low']) / 2
    diff = data['Close'] - M
    R = data['High'] - data['Low']
    smooth_diff = diff.ewm(span=smoothing, adjust=False).mean().ewm(span=smoothing, adjust=False).mean()
    smooth_R = R.ewm(span=smoothing, adjust=False).mean().ewm(span=smoothing, adjust=False).mean()
    smi = 100 * (smooth_diff / (0.5 * smooth_R)).replace([np.inf, -np.inf], 0)
    smi_d = smi.rolling(window=d_period).mean()
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

def get_stock_data(ticker):
    # Set auto_adjust to False to avoid warning
    stock_data = yf.download(ticker, period='1y', interval='1d', auto_adjust=False)
    return stock_data

def analyze(data):
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    calculate_ichimoku(data)
    fib_levels = calculate_fibonacci_levels(data)
    vp_peak, vp_hist, vp_bins = calculate_volume_profile(data, bins=20)
    smi, smi_d = calculate_smi(data)
    div_series = compute_divergence_series(data)
    div_series = pd.to_numeric(div_series, errors='coerce')

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
    raw_RSI = 50 - latest_rsi
    score_RSI = WEIGHT_RSI * raw_RSI

    # 2. MACD: percentage difference between MACD and Signal
    raw_MACD = (latest_macd - latest_signal) / latest_close * 100
    score_MACD = WEIGHT_MACD * raw_MACD

    # Compute slopes for MACD and Signal using MACD_SLOPE_WINDOW
    if len(data) >= MACD_SLOPE_WINDOW:
        x = np.arange(MACD_SLOPE_WINDOW)
        macd_values = data['MACD'].iloc[-MACD_SLOPE_WINDOW:].values
        signal_values = data['Signal'].iloc[-MACD_SLOPE_WINDOW:].values
        macd_slope = float(np.polyfit(x, macd_values, 1)[0])
        signal_slope = float(np.polyfit(x, signal_values, 1)[0])
    else:
        macd_slope = 0.0
        signal_slope = 0.0

    # 3. Bollinger Bands: deviation of current price from SMA relative to band width
    latest_SMA = (latest_upper_band + latest_lower_band) / 2
    band_width = latest_upper_band - latest_lower_band
    raw_BB = (latest_SMA - latest_close) / band_width if band_width != 0 else 0
    score_BB = WEIGHT_BB * raw_BB

    # 4. Ichimoku: ratio difference between current price and mid value
    mid_value = (latest_senkou_span_a + latest_senkou_span_b) / 2
    raw_Ichi = (latest_close - mid_value) / mid_value
    score_Ichi = WEIGHT_ICHIMOKU * raw_Ichi

    # 5. Divergence: based on RSI and price change
    if latest_close != 0:
        if latest_rsi <= 50:
            raw_div = (50 - latest_rsi) * ((latest_close - previous_close) / latest_close) * 5
        else:
            raw_div = - (latest_rsi - 50) * ((latest_close - previous_close) / latest_close) * 5
    else:
        raw_div = 0
    score_div = WEIGHT_DIV * raw_div

    # 6. Volume Profile: relative difference between current price and vp_peak
    distance = (latest_close - vp_peak) / vp_peak
    if distance > 0:
        raw_VP = - distance * 100
    elif distance < 0:
        raw_VP = abs(distance) * 100
    else:
        raw_VP = 0
    score_VP = WEIGHT_VP * raw_VP

    # 7. Volume Ratio: based on ratio of today's volume to 20-day average volume
    avg_volume = np.nanmean(data['Volume'].rolling(window=20).mean().values)
    volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1
    if latest_close < previous_close:
        raw_Vol = - (volume_ratio - 1) * 10
    elif latest_close > previous_close:
        raw_Vol = (volume_ratio - 1) * 10
    else:
        raw_Vol = 0
    score_Vol = WEIGHT_VOL * raw_Vol

    # 8. SMI Signal: using slopes over recent window (with SMI_SLOPE_WINDOW)
    if len(smi) >= SMI_SLOPE_WINDOW:
        x = np.arange(SMI_SLOPE_WINDOW)
        k_values = smi.iloc[-SMI_SLOPE_WINDOW:].values
        d_values = smi_d.iloc[-SMI_SLOPE_WINDOW:].values
        k_slope = float(np.polyfit(x, k_values, 1)[0])
        d_slope = float(np.polyfit(x, d_values, 1)[0])
        slope_diff = k_slope - d_slope
    else:
        k_slope = 0.0
        d_slope = 0.0
        slope_diff = 0.0
    score_SMI = WEIGHT_SMI * slope_diff

    total_score = score_RSI + score_MACD + score_BB + score_Ichi + score_div + score_VP + score_Vol + score_SMI
    final_score = total_score #100 * np.tanh(total_score / 20)

    details = {
        "RSI": {"raw": round(raw_RSI, 2), "score": round(score_RSI, 2)},
        "MACD": {
            "raw": round(raw_MACD, 2),
            "score": round(score_MACD, 2),
            "macd": round(latest_macd, 2),
            "signal": round(latest_signal, 2),
            "macd_slope": round(macd_slope, 2),
            "signal_slope": round(signal_slope, 2)
        },
        "BollingerBands": {"raw": round(raw_BB, 2), "score": round(score_BB, 2)},
        "Ichimoku": {"raw": round(raw_Ichi, 2), "score": round(score_Ichi, 2)},
        "Divergence": {"raw": round(raw_div, 2), "score": round(score_div, 2)},
        "VolumeProfile": {"raw": round(raw_VP, 2), "score": round(score_VP, 2), "vp_peak": round(vp_peak, 2)},
        "VolumeRatio": {"raw": round(raw_Vol, 2), "score": round(score_Vol, 2)},
        "SMI": {
            "raw": round(slope_diff, 2),
            "score": round(score_SMI, 2),
            "K": round(latest_smi, 2),
            "D": round(latest_smi_d, 2),
            "K_slope": round(k_slope, 2),
            "D_slope": round(d_slope, 2)
        }
    }

    return {
        "date": latest_date,
        "close": f"{latest_close:.2f}",
        "trend_score": f"{final_score:.2f}",
        "details": details
    }

@router.get("/{ticker}/divergence", response_model=DivergenceResponse)
async def analyze_stock(ticker: str):
    try:
        data = get_stock_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    analysis_results = analyze(data)

    return {
        "ticker": ticker,
        "date": analysis_results['date'],
        "current_price": analysis_results['close'],
        "trend_score": analysis_results['trend_score'],
        "details": analysis_results['details']
    }

def analyze_all(data):
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    calculate_ichimoku(data)
    smi, smi_d = calculate_smi(data)

    trend_scores = {}

    for i in range(1, len(data)):  # 날짜별로 trend_score 계산
        try:
            # 🎯 NaN 처리 및 단일 숫자로 변환
            latest_close = float(np.nan_to_num(data['Close'].iloc[i], nan=0.0))
            latest_rsi = float(np.nan_to_num(data['RSI'].iloc[i], nan=50.0))
            latest_macd = float(np.nan_to_num(data['MACD'].iloc[i], nan=0.0))
            latest_signal = float(np.nan_to_num(data['Signal'].iloc[i], nan=0.0))
            latest_upper_band = float(np.nan_to_num(data['Upper Band'].iloc[i], nan=latest_close))
            latest_lower_band = float(np.nan_to_num(data['Lower Band'].iloc[i], nan=latest_close))
            latest_senkou_span_a = float(np.nan_to_num(data['Senkou Span A'].iloc[i], nan=latest_close))
            latest_senkou_span_b = float(np.nan_to_num(data['Senkou Span B'].iloc[i], nan=latest_close))
            latest_volume = float(np.nan_to_num(data['Volume'].iloc[i], nan=1.0))
            latest_smi = float(np.nan_to_num(smi.iloc[i], nan=0.0))
            latest_smi_d = float(np.nan_to_num(smi_d.iloc[i], nan=0.0))
            previous_close = float(np.nan_to_num(data['Close'].iloc[i - 1], nan=latest_close))

            # RSI 신호
            raw_RSI = float(50 - latest_rsi)  # 🎯 float 변환 추가
            score_RSI = WEIGHT_RSI * raw_RSI

            # MACD 신호
            raw_MACD = float((latest_macd - latest_signal) / latest_close * 100) if latest_close != 0 else 0.0
            score_MACD = WEIGHT_MACD * raw_MACD

            # 볼린저 밴드
            latest_SMA = float((latest_upper_band + latest_lower_band) / 2)
            band_width = float(latest_upper_band - latest_lower_band)
            raw_BB = float((latest_SMA - latest_close) / band_width) if band_width != 0 else 0.0
            score_BB = WEIGHT_BB * raw_BB

            # 이치모쿠
            mid_value = float((latest_senkou_span_a + latest_senkou_span_b) / 2)
            raw_Ichi = float((latest_close - mid_value) / mid_value) if mid_value != 0 else 0.0
            score_Ichi = WEIGHT_ICHIMOKU * raw_Ichi

            # 다이버전스
            if latest_close != 0:
                if latest_rsi <= 50:
                    raw_div = float((50 - latest_rsi) * ((latest_close - previous_close) / latest_close) * 5)
                else:
                    raw_div = float(- (latest_rsi - 50) * ((latest_close - previous_close) / latest_close) * 5)
            else:
                raw_div = 0.0
            score_div = WEIGHT_DIV * raw_div

            # ✅ 최종 스코어 계산 (반드시 float 변환)
            total_score = float(score_RSI + score_MACD + score_BB + score_Ichi + score_div)
            final_score = round(total_score, 2)

            # NaN 체크 및 변환
            if pd.isna(final_score) or np.isnan(final_score):
                final_score = 0.0  # NaN이 있으면 0으로 변환

            # 📌 trend_scores를 딕셔너리로 저장
            trend_scores[data.index[i].strftime("%Y-%m-%d")] = final_score

        except Exception as e:
            print(f"Error processing {data.index[i]}: {e}")  # 디버깅 로그

    return trend_scores




@router.get("/{ticker}/divergence_all", response_model=TrendScoreResponse)
async def analyze_stock_all(ticker: str):
    try:
        data = get_stock_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

    trend_scores = analyze_all(data)

    return {"ticker": ticker, "trend_scores": trend_scores}