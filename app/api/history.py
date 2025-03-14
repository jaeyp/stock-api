from fastapi import APIRouter, HTTPException
from app.models.history import HistoryResponse, HistoryData
import yfinance as yf
import pandas as pd

router = APIRouter()

@router.get("/{ticker}/history", response_model=HistoryResponse)
async def get_stock_history(ticker: str):
    try:
        # Fetch historical stock data for the last year
        stock_data = yf.download(ticker, period='1y', interval='1d', auto_adjust=True)

        # Debug: Print the fetched stock data
        print("=== Stock Data Before Reset Index ===")
        print(stock_data.head())  # Check DataFrame structure
        
        # Check if data is empty
        if stock_data is None or stock_data.empty:
            raise HTTPException(status_code=404, detail="No historical data found for the given ticker.")

        # Drop rows with NaN values
        stock_data = stock_data.dropna()

        # Reset index and flatten MultiIndex column names
        stock_data.reset_index(inplace=True)

        # Fix MultiIndex column names
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['Date' if col[0] == 'Date' else col[0] for col in stock_data.columns]

        # Debug: Check column names
        print("=== Flattened Columns ===")
        print(stock_data.columns)

        # Ensure "Date" is formatted correctly
        if "Date" in stock_data.columns and pd.api.types.is_datetime64_any_dtype(stock_data["Date"]):
            stock_data["Date"] = stock_data["Date"].dt.strftime("%Y-%m-%d")

        # Debug: Check final structure before processing
        print("=== Final DataFrame Structure ===")
        print(stock_data.head())

        # Convert the DataFrame to a list of HistoryData
        history = [
            HistoryData(
                Date=row.Date,
                Open=row.Open,  
                High=row.High,
                Low=row.Low,
                Close=row.Close,
                Volume=row.Volume
            )
            for row in stock_data.itertuples(index=False)
        ]

        return {
            "ticker": ticker,
            "history": history
        }

    except Exception as e:
        print(f"Error fetching historical data: {e}")  # Log error to terminal
        raise HTTPException(status_code=400, detail=f"Error fetching historical data for {ticker}: {str(e)}")
