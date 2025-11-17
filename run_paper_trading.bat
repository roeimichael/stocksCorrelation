@echo off
REM Daily Paper Trading Runner for Windows
REM Run this script once per day after market close

echo ==============================================
echo SP500-ANALOGS PAPER TRADING
echo %date% %time%
echo ==============================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv .venv
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Run paper trading
echo.
echo [2/2] Running paper trading...
python scripts\paper_trade_daily.py

echo.
echo ==============================================
echo PAPER TRADING COMPLETE!
echo ==============================================
echo.
echo To view results:
echo   - Check console output above
echo   - View CSV files in results\paper_trading\
echo   - Generate Excel report: python scripts\export_to_excel.py
echo.
pause
