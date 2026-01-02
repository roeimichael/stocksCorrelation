#!/usr/bin/env python3
"""Test different yfinance API configurations to find what works."""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

print("="*80)
print("YFINANCE API TESTING")
print("="*80)

# Test symbol
symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2026-01-02"

print(f"\nTest symbol: {symbol}")
print(f"Date range: {start_date} to {end_date}")
print(f"yfinance version: {yf.__version__}")
print("="*80)

# Test 1: yf.download() - default parameters
print("\n[TEST 1] yf.download() - default parameters")
try:
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Sample row:\n{data.head(1)}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: yf.download() - with show_errors=False
print("\n[TEST 2] yf.download() - with show_errors=False")
try:
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, show_errors=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 3: yf.download() - with auto_adjust=True
print("\n[TEST 3] yf.download() - with auto_adjust=True")
try:
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 4: yf.download() - with auto_adjust=False
print("\n[TEST 4] yf.download() - with auto_adjust=False")
try:
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 5: ticker.history() - default parameters
print("\n[TEST 5] ticker.history() - default parameters")
try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 6: ticker.history() - with auto_adjust=True, actions=True
print("\n[TEST 6] ticker.history() - auto_adjust=True, actions=True")
try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, auto_adjust=True, actions=True)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 7: ticker.history() - with auto_adjust=False, actions=False
print("\n[TEST 7] ticker.history() - auto_adjust=False, actions=False")
try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 8: ticker.history() - with period instead of dates
print("\n[TEST 8] ticker.history() - using period='1mo'")
try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1mo")
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 9: ticker.history() - with period='max'
print("\n[TEST 9] ticker.history() - using period='max'")
try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 10: yf.download() with interval parameter
print("\n[TEST 10] yf.download() - with interval='1d'")
try:
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 11: yf.download() - shorter date range (last 30 days)
recent_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
print(f"\n[TEST 11] yf.download() - shorter range ({recent_start} to {end_date})")
try:
    data = yf.download(symbol, start=recent_start, end=end_date, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 12: Test with datetime objects instead of strings
print("\n[TEST 12] yf.download() - using datetime objects")
try:
    from datetime import datetime
    start_dt = datetime(2023, 1, 1)
    end_dt = datetime(2026, 1, 2)
    data = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 13: Multiple symbols at once
print("\n[TEST 13] yf.download() - multiple symbols")
try:
    data = yf.download("AAPL MSFT", start=recent_start, end=end_date, progress=False)
    print(f"  ✓ Success: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
    if not data.empty:
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 14: Check ticker info
print("\n[TEST 14] ticker.info - check if ticker is valid")
try:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    print(f"  ✓ Success: Got ticker info")
    print(f"  Symbol: {info.get('symbol', 'N/A')}")
    print(f"  Name: {info.get('longName', 'N/A')}")
    print(f"  Exchange: {info.get('exchange', 'N/A')}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
print("\nReview the results above to see which method successfully retrieves data.")
print("Look for tests with '✓ Success' and non-zero row counts.")
