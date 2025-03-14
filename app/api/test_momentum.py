import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
import io
import base64
import matplotlib.dates as mdates  # 날짜 포맷을 위한 모듈
from app.api.momentum import analyze_all, get_stock_data  # analyze_all 사용

router = APIRouter()

@router.get("/{ticker}/test")
async def get_stock_graph(ticker: str, period: str = '1y'):
    try:
        # 📌 1. Ticker 데이터 가져오기
        data = get_stock_data(ticker, period)
        if data.empty:
            raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

        # 📌 2. 분석 실행
        history = analyze_all(data)

        # 📌 3. Close와 Momentum Strength 데이터 변환
        close_prices = pd.Series(history["close"])
        momentum_strength = pd.Series(history["momentum_strength"])

        # ✅ 📌 4. 날짜 인덱스를 `datetime` 형식으로 변환 (1970-01 방지)
        close_prices.index = pd.to_datetime(close_prices.index)
        momentum_strength.index = pd.to_datetime(momentum_strength.index)

        # 📌 5. Close 값을 -100 ~ 100 범위로 변환 (정규화)
        min_close, max_close = close_prices.min(), close_prices.max()
        scaled_close = ((close_prices - min_close) / (max_close - min_close)) * 200 - 100

        # 📌 6. 차이가 100 이상 나는 지점 찾기
        large_diff_mask = np.abs(scaled_close - momentum_strength) >= 100
        large_diff_dates = close_prices.index[large_diff_mask]
        large_diff_values = scaled_close[large_diff_mask]

        # 📌 7. 그래프 생성
        plt.figure(figsize=(20, 8))
        plt.plot(close_prices.index, scaled_close, label="Normalized Close Prices", color="blue")  # 변환된 close
        plt.plot(momentum_strength.index, momentum_strength, label="Momentum Strength", color="red", linewidth=1)

        # ✅ 📌 8. 차이가 100 이상인 지점에 'X' 표시
        plt.scatter(large_diff_dates, large_diff_values, color='black', marker='x', s=70, label="High Difference")

        # 📌 9. x축 날짜를 "YYYY-MM" 형식으로 변경
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # 년-월 포맷 적용
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1개월 간격으로 표시

        # ✅ 10. Y축 20 단위로 고정 (100, 80, 60, ..., -100)
        plt.yticks(np.arange(-100, 101, 20))  # -100에서 100까지 20 간격

        plt.title(f"{ticker} - Stock Prices & Momentum Strength")
        plt.xlabel("Date")
        plt.ylabel("Scaled Value (-100 to 100)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 그리드 추가

        # 📌 11. x축 라벨 회전 및 간격 조절
        plt.xticks(rotation=45, ha="right")  # 45도 회전, 오른쪽 정렬
        plt.tight_layout()  # 자동 간격 조정

        # 📌 12. 그래프를 Base64로 변환 후 반환
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image": f"data:image/png;base64,{image_base64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")
