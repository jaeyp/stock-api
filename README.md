# Stock API

A FastAPI-based stock data API server that provides real-time stock information using yfinance and finvizfinance.

## Features

- Stock basic information retrieval (company name, current price, market cap)
- 5-day historical stock price data
- CORS support for Vue.js frontend integration

## Tech Stack

- FastAPI
- yfinance (Yahoo Finance data)
- finvizfinance (Finviz data)
- Python 3.8+

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stock-api.git
cd stock-api
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Unix/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app.main:app --reload
```

The server will run at http://localhost:8000

## API Endpoints

### GET /stocks/{ticker}

Retrieves information for a specific stock.

**Parameters:**
- `ticker`: Stock symbol (e.g., AAPL, GOOGL, MSFT)

**Response Example:**
```json
{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "current_price": 182.52,
  "market_cap": 2825262841856,
  "history": [
    {
      "Open": 181.27,
      "High": 182.93,
      "Low": 180.88,
      "Close": 182.52,
      "Volume": 60893274,
      "Dividends": 0.0,
      "StockSplits": 0.0,
      "Date": "2024-02-23"
    }
  ]
}
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Setup

1. VS Code Configuration
   - Install Python extension
   - Install Pylance extension
   - Set Python interpreter to virtual environment

2. Code Formatting
   - Use black for Python code formatting
   - Use isort for import statement sorting

## License

This project is licensed under the MIT License.
