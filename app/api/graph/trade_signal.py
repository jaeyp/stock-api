import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
import io
import base64
import matplotlib.dates as mdates  # 날짜 포맷을 위한 모듈
from ..momentum import analyze_all, get_stock_data  # analyze_all 사용

DEFAULT_SIGNAL_THRESHOLD = 100
AGGRESSIVE_BUY_THRESHOLD = 95
AGGRESSIVE_SELL_THRESHOLD = 105

router = APIRouter()

@router.get("/{ticker}/chart/trade_signal")
async def get_stock_graph(ticker: str, period: str = '1y', mode: str = "conservative"):
    try:
        # 1. Ticker 데이터 가져오기
        data = get_stock_data(ticker, period)
        if data.empty:
            raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

        # 2. 분석 실행
        history = analyze_all(data, mode)

        # 3. Close와 Momentum Strength 데이터 변환
        close_prices = pd.Series(history["close"])
        momentum_strength = pd.Series(history["momentum_strength"])

        # 4. 날짜 인덱스를 `datetime` 형식으로 변환 (1970-01 방지)
        close_prices.index = pd.to_datetime(close_prices.index)
        momentum_strength.index = pd.to_datetime(momentum_strength.index)

        # 5. Close 값을 -100 ~ 100 범위로 변환 (정규화)
        min_close, max_close = close_prices.min(), close_prices.max()
        scaled_close = ((close_prices - min_close) / (max_close - min_close)) * 200 - 100

        # ✅ Aggressive 모드에서 매수·매도 신호 threshold 조정
        buy_threshold = np.where(mode == "aggressive", AGGRESSIVE_BUY_THRESHOLD, DEFAULT_SIGNAL_THRESHOLD)
        sell_threshold = np.where(mode == "aggressive", AGGRESSIVE_SELL_THRESHOLD, DEFAULT_SIGNAL_THRESHOLD)

        # 6. 차이가 buy_threshold 또는 sell_threshold 이상 나는 지점 찾기
        diff_values = scaled_close - momentum_strength  # 차이 계산
        large_diff_mask = (diff_values >= sell_threshold) | (diff_values <= -buy_threshold)
        large_diff_dates = close_prices.index[large_diff_mask]
        large_diff_values = scaled_close[large_diff_mask]

        # 7. 차이가 큰 지점에서 Momentum Strength와 비교하여 마커 설정
        is_below = momentum_strength[large_diff_mask] < scaled_close[large_diff_mask]  # Close가 Momentum보다 높음
        is_above = ~is_below  # 반대

        # 8. 그래프 생성
        plt.figure(figsize=(20, 8))
        plt.plot(close_prices.index, scaled_close, label="Normalized Close Prices", color="green")  # 변환된 close
        plt.plot(momentum_strength.index, momentum_strength, label="Momentum Strength", color="red", linewidth=1)

        # 9. 차이가 buy_threshold 또는 sell_threshold 이상인 지점에 마커 표시
        plt.scatter(
            large_diff_dates[is_below], large_diff_values[is_below], 
            color='#3a86ff', marker='o', s=100, label="Sell Signal"
        )  # 동그라미 (파란색)
        plt.scatter(
            large_diff_dates[is_above], large_diff_values[is_above], 
            color='orange', marker='D', s=90, label="Buy Signal"
        )  # 다이아몬드 (주황색)

        # 10. 마커 **하단**에 Close Price 표시 (검은색)
        texts = [f"{price:.2f}" for price in close_prices[large_diff_mask]]
        for i, txt in enumerate(texts):
            plt.text(large_diff_dates[i], large_diff_values[i] - 6, txt, 
                     fontsize=9, ha='center', color='black')

        # 11. 마커 **상단**에 Close - Trend Score 차이(diff) 값 표시 (양수=파란색, 음수=빨간색)
        for i, diff in enumerate(diff_values[large_diff_mask]):
            color = 'blue' if diff > 0 else 'red'  # 양수는 파란색, 음수는 빨간색
            plt.text(large_diff_dates[i], large_diff_values[i] + 3, f"{abs(diff):.2f}", 
                    fontsize=9, ha='center', color=color)

        # 12. Close Price의 시작과 끝에 가격 표시
        first_date, last_date = close_prices.index[0], close_prices.index[-1]
        first_price, last_price = scaled_close.iloc[0], scaled_close.iloc[-1]

        plt.text(first_date, first_price, f"{close_prices.iloc[0]:.2f}", 
                fontsize=10, ha='right', va='bottom', color='black')
        plt.text(last_date, last_price, f"{close_prices.iloc[-1]:.2f}", 
                fontsize=10, ha='left', va='bottom', color='black')
        
        # 12-1. 마지막 날짜의 Close - Trend Score 차이(diff) 값도 표시
        last_date_diff = scaled_close.iloc[-1] - momentum_strength.iloc[-1]
        last_date_diff_color = 'blue' if last_date_diff > 0 else 'red'

        plt.text(last_date, last_price + 6, f"{abs(last_date_diff):.2f}", 
                fontsize=10, ha='left', color=last_date_diff_color)
        
        # ✅ 12-2. Trend Score의 시작과 끝 값도 표시
        first_momentum, last_momentum = momentum_strength.iloc[0], momentum_strength.iloc[-1]

        plt.text(first_date, first_momentum, f"{first_momentum:.2f}", 
                fontsize=10, ha='right', va='bottom', color='black')  # 빨간색 (Trend Score)
        plt.text(last_date, last_momentum, f"{last_momentum:.2f}", 
                fontsize=10, ha='left', va='bottom', color='black')  # 빨간색 (Trend Score)

        # 13. x축 날짜를 "YYYY-MM" 형식으로 변경
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # 년-월 포맷 적용
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1개월 간격으로 표시

        # 14. Y축 20 단위로 고정 (100, 80, 60, ..., -100)
        plt.yticks(np.arange(-100, 101, 20))  # -100에서 100까지 20 간격

        plt.title(f"{ticker} - Stock Prices & Momentum Strength")
        plt.xlabel("Date")
        plt.ylabel("Scaled Value (-100 to 100)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 그리드 추가

        # 15. x축 라벨 회전 및 간격 조절
        plt.xticks(rotation=45, ha="right")  # 45도 회전, 오른쪽 정렬
        plt.tight_layout()  # 자동 간격 조정

        # 16. 그래프를 Base64로 변환 후 반환
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        return {"image": f"data:image/png;base64,{image_base64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")
