import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
import io
import base64
import matplotlib.dates as mdates
from datetime import datetime
from ..momentum import analyze_all, get_stock_data2

CONSERVATIVE_BUY_THRESHOLD = 100
CONSERVATIVE_SELL_THRESHOLD = 110
AGGRESSIVE_BUY_THRESHOLD = 100
AGGRESSIVE_SELL_THRESHOLD = 110

router = APIRouter()

@router.get("/{ticker}/chart/trade_signal")
async def get_stock_graph(ticker: str, period: str = '1y', mode: str = "conservative", reference_date = None): # 2025/03/18 or 2025-03-18
    try:
        # 1. Get ticker data
        data = get_stock_data2(ticker, period, reference_date)
        if data.empty:
            raise HTTPException(status_code=400, detail="No data fetched for the given ticker.")

        # 2. Run analysis
        history = analyze_all(data, mode)

        # 3. Convert Close and Momentum Strength data
        close_prices = pd.Series(history["close"])
        momentum_strength = pd.Series(history["momentum_strength"])

        # 4. Convert date index to datetime format (to prevent 1970-01 issues)
        close_prices.index = pd.to_datetime(close_prices.index)
        momentum_strength.index = pd.to_datetime(momentum_strength.index)

        # 5. Normalize Close values to the range -100 ~ 100
        min_close, max_close = close_prices.min(), close_prices.max()
        scaled_close = ((close_prices - min_close) / (max_close - min_close)) * 200 - 100

        # In aggressive mode, adjust buy/sell signal thresholds
        buy_threshold = np.where(mode == "aggressive", AGGRESSIVE_BUY_THRESHOLD, CONSERVATIVE_BUY_THRESHOLD)
        sell_threshold = np.where(mode == "aggressive", AGGRESSIVE_SELL_THRESHOLD, CONSERVATIVE_SELL_THRESHOLD)

        # 6. Find points where the difference is greater than buy_threshold or sell_threshold
        diff_values = scaled_close - momentum_strength  # Calculate difference
        large_diff_mask = (diff_values >= sell_threshold) | (diff_values <= -buy_threshold)
        large_diff_dates = close_prices.index[large_diff_mask]
        large_diff_values = scaled_close[large_diff_mask]

        # 7. Set markers on points with large differences comparing Momentum Strength
        is_below = momentum_strength[large_diff_mask] < scaled_close[large_diff_mask]  # Close is higher than Momentum
        is_above = ~is_below  # Opposite

        # 8. Create the graph
        plt.figure(figsize=(20, 8))
        plt.plot(close_prices.index, scaled_close, label="Normalized Close Prices", color="green")  # Normalized close
        plt.plot(momentum_strength.index, momentum_strength, label="Momentum Strength", color="red", linewidth=1)

        # 9. Place markers on points with difference above threshold (or below threshold)
        """ plt.scatter(
            large_diff_dates[is_below], large_diff_values[is_below], 
            color='#3a86ff', marker='o', s=100, label="Sell Signal"
        )  # Circle (blue)
        plt.scatter(
            large_diff_dates[is_above], large_diff_values[is_above], 
            color='orange', marker='D', s=90, label="Buy Signal"
        )  # Diamond (orange) """
        plt.scatter(
            large_diff_dates[is_below], large_diff_values[is_below],
            color='#1e88e5', marker='o', s=100, label="Sell Signal"
        )
        plt.scatter(
            large_diff_dates[is_above], large_diff_values[is_above],
            color='#f77f00', marker='D', s=90, label="Buy Signal"
        )

        # 10. Display Close Price below each marker (in black)
        texts = [f"{price:.2f}" for price in close_prices[large_diff_mask]]
        for i, txt in enumerate(texts):
            plt.text(large_diff_dates[i], large_diff_values[i] - 6, txt, 
                     fontsize=9, ha='center', color='black')

        # 11. Display the difference (Close - Trend Score) above each marker 
        # (blue for positive, red for negative)
        for i, diff in enumerate(diff_values[large_diff_mask]):
            color = 'blue' if diff > 0 else 'red'
            plt.text(large_diff_dates[i], large_diff_values[i] + 3, f"{abs(diff):.2f}", 
                    fontsize=9, ha='center', color=color)

        # 12. Display the price at the beginning and end of Close Price
        first_date, last_date = close_prices.index[0], close_prices.index[-1]
        first_price, last_price = scaled_close.iloc[0], scaled_close.iloc[-1]

        plt.text(first_date, first_price, f"{close_prices.iloc[0]:.2f}", 
                fontsize=10, ha='right', va='bottom', color='black')
        plt.text(last_date, last_price, f"{close_prices.iloc[-1]:.2f}", 
                fontsize=10, ha='left', va='bottom', color='black')
        
        # 12-1. Also display the difference (Close - Trend Score) at the last date
        last_date_diff = scaled_close.iloc[-1] - momentum_strength.iloc[-1]
        last_date_diff_color = 'blue' if last_date_diff > 0 else 'red'

        plt.text(last_date, last_price + 6, f"{abs(last_date_diff):.2f}", 
                fontsize=10, ha='left', color=last_date_diff_color)
        
        # 12-2. Also display the starting and ending Trend Score values
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
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")
