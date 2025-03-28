import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
import io
import base64
import matplotlib.dates as mdates
from datetime import datetime
from ..momentum import analyze_all, get_stock_data2

CONSERVATIVE_BUY_THRESHOLD = 110
AGGRESSIVE_BUY_THRESHOLD = 100
CONSERVATIVE_SELL_THRESHOLD = 110
AGGRESSIVE_SELL_THRESHOLD = 100

router = APIRouter()

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

@router.get("/{ticker}/chart/trade_signal")
async def get_stock_graph(ticker: str, period: str = '6mo', mode: str = "conservative", reference_date=None):
    try:
        # 1. Get ticker data
        # Download data for twice the period to obtain proper historical data for analysis.
        extended_period = double_period(period)
        data = get_stock_data2(ticker, extended_period, reference_date)

        if data.empty:
            raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

        # 2. Run analysis
        history = analyze_all(data, mode)

        # 3. Convert Close and Momentum Strength data
        close_prices = pd.Series(history["close"])
        momentum_strength = pd.Series(history["momentum_strength"])

        # 4. Convert date index to datetime format
        close_prices.index = pd.to_datetime(close_prices.index)
        momentum_strength.index = pd.to_datetime(momentum_strength.index)

        # 5. Normalize Close values to the range -100 ~ 100
        min_close, max_close = close_prices.min(), close_prices.max()
        scaled_close = ((close_prices - min_close) / (max_close - min_close)) * 200 - 100

        # 6. Calculate difference (Scaled Close - Momentum Strength)
        diff_values = scaled_close - momentum_strength

        # 7. Define masks for Buy and Sell signals using both thresholds
        buy_light_mask = (diff_values >= -CONSERVATIVE_BUY_THRESHOLD) & (diff_values <= -AGGRESSIVE_BUY_THRESHOLD)
        buy_extreme_mask = diff_values < -CONSERVATIVE_BUY_THRESHOLD

        sell_light_mask = (diff_values >= AGGRESSIVE_SELL_THRESHOLD) & (diff_values <= CONSERVATIVE_SELL_THRESHOLD)
        sell_extreme_mask = diff_values > CONSERVATIVE_SELL_THRESHOLD

        # 8. Create the graph
        plt.figure(figsize=(20, 8))
        plt.plot(close_prices.index, scaled_close, label="Normalized Close Prices", color="green")
        plt.plot(momentum_strength.index, momentum_strength, label="Momentum Strength", color="red", linewidth=1)

        # 9. Place markers for each condition
        # Buy signals:
        plt.scatter(
            close_prices.index[buy_light_mask],
            scaled_close[buy_light_mask],
            color="#ffd8a8",  # light orange
            marker="D",       # circle
            s=90,
            label="Buy Signal (Light)"
        )
        plt.scatter(
            close_prices.index[buy_extreme_mask],
            scaled_close[buy_extreme_mask],
            color="#fb5607",   # deep orange
            marker="D",       # diamond
            s=90,
            label="Buy Signal (Extreme)"
        )
        # Sell signals:
        plt.scatter(
            close_prices.index[sell_light_mask],
            scaled_close[sell_light_mask],
            color="#aedff7",  # light blue
            marker="o",       # diamond
            s=100,
            label="Sell Signal (Light)"
        )
        plt.scatter(
            close_prices.index[sell_extreme_mask],
            scaled_close[sell_extreme_mask],
            color="#023e8a",  # deep blue
            marker="o",       # circle
            s=100,
            label="Sell Signal (Extreme)"
        )

        # 10. Display Close Price below each marker (in black)
        for mask in [buy_light_mask, buy_extreme_mask, sell_light_mask, sell_extreme_mask]:
            group_dates = close_prices.index[mask]
            group_values = scaled_close[mask]
            group_texts = [f"{price:.2f}" for price in close_prices[mask]]
            for i, txt in enumerate(group_texts):
                plt.text(group_dates[i], group_values[i] - 6, txt,
                         fontsize=9, ha='center', color='black')

        # 11. Display the difference (Close - Momentum Strength) above each marker 
        for mask in [buy_light_mask, buy_extreme_mask, sell_light_mask, sell_extreme_mask]:
            group_dates = close_prices.index[mask]
            group_values = scaled_close[mask]
            group_diffs = diff_values[mask]
            for i, diff in enumerate(group_diffs):
                color = 'blue' if diff > 0 else 'red'
                plt.text(group_dates[i], group_values[i] + 3, f"{abs(diff):.2f}",
                         fontsize=9, ha='center', color=color)

        # 12. Display the price at the beginning and end of Close Price
        first_date, last_date = close_prices.index[0], close_prices.index[-1]
        first_price, last_price = scaled_close.iloc[0], scaled_close.iloc[-1]
        plt.text(first_date, first_price, f"{close_prices.iloc[0]:.2f}",
                 fontsize=10, ha='right', va='bottom', color='black')
        plt.text(last_date, last_price, f"{close_prices.iloc[-1]:.2f}",
                 fontsize=10, ha='left', va='bottom', color='black')

        # 12-1. Also display the difference (Close - Momentum Strength) at the last date
        last_date_diff = scaled_close.iloc[-1] - momentum_strength.iloc[-1]
        last_date_diff_color = 'blue' if last_date_diff > 0 else 'red'
        plt.text(last_date, last_price + 6, f"{abs(last_date_diff):.2f}",
                 fontsize=10, ha='left', color=last_date_diff_color)

        # 12-2. Also display the starting and ending Momentum Strength values
        first_momentum, last_momentum = momentum_strength.iloc[0], momentum_strength.iloc[-1]
        plt.text(first_date, first_momentum, f"{first_momentum:.2f}",
                 fontsize=10, ha='right', va='bottom', color='black')
        plt.text(last_date, last_momentum, f"{last_momentum:.2f}",
                 fontsize=10, ha='left', va='bottom', color='black')

        # 13. Change x-axis date format to "YYYY-MM"
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # 14. Set Y-axis ticks at 20 unit intervals (-100 to 100)
        plt.yticks(np.arange(-100, 101, 20))

        plt.title(f"{ticker} - Stock Prices & Momentum Strength")
        plt.xlabel("Date")
        plt.ylabel("Scaled Value (-100 to 100)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # 15. Rotate and adjust x-axis labels
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 16. Convert the graph to Base64 and return
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image": f"data:image/png;base64,{image_base64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph for {ticker}: {str(e)}")
