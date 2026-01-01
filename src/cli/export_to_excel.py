"""Export paper trading results to Excel for monthly review."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger import get_logger


logger = get_logger(__name__)


def export_to_excel(output_file: str | None = None) -> None:
    """Export all paper trading results to a single Excel file with multiple sheets."""
    logger.info("Exporting paper trading results to Excel...")

    # Determine output file
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'results/paper_trading/paper_trading_report_{timestamp}.xlsx'

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path('results/paper_trading')

    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Daily P&L
        daily_file = data_dir / 'daily_pnl.csv'
        if daily_file.exists():
            daily_df = pd.read_csv(daily_file)
            daily_df['date'] = pd.to_datetime(daily_df['date'])

            # Add daily change column
            daily_df['daily_pnl_change'] = daily_df['total_pnl'].diff()

            daily_df.to_excel(writer, sheet_name='Daily PnL', index=False)
            logger.info(f"Exported daily P&L: {len(daily_df)} days")
        else:
            logger.warning("No daily P&L file found")

        # Sheet 2: Closed Trades
        trades_file = data_dir / 'closed_trades.csv'
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

            # Calculate hold duration
            trades_df['hold_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days

            trades_df.to_excel(writer, sheet_name='Closed Trades', index=False)
            logger.info(f"Exported closed trades: {len(trades_df)} trades")
        else:
            logger.warning("No closed trades file found")
            trades_df = None

        # Sheet 3: Monthly Summary
        if daily_file.exists():
            daily_df['month'] = daily_df['date'].dt.to_period('M')

            monthly_summary = daily_df.groupby('month').agg({
                'total_pnl': 'last',
                'portfolio_value': 'last',
                'total_return_pct': 'last',
                'num_positions': 'mean'
            }).reset_index()

            monthly_summary['month'] = monthly_summary['month'].astype(str)
            monthly_summary['monthly_pnl'] = monthly_summary['total_pnl'].diff()

            monthly_summary.to_excel(writer, sheet_name='Monthly Summary', index=False)
            logger.info(f"Exported monthly summary: {len(monthly_summary)} months")

        # Sheet 4: Trade Statistics
        if trades_df is not None and len(trades_df) > 0:
            stats = {
                'Total Trades': len(trades_df),
                'Winning Trades': len(trades_df[trades_df['realized_pnl'] > 0]),
                'Losing Trades': len(trades_df[trades_df['realized_pnl'] <= 0]),
                'Win Rate (%)': (len(trades_df[trades_df['realized_pnl'] > 0]) / len(trades_df)) * 100,
                'Average P&L': trades_df['realized_pnl'].mean(),
                'Average Win': trades_df[trades_df['realized_pnl'] > 0]['realized_pnl'].mean() if len(trades_df[trades_df['realized_pnl'] > 0]) > 0 else 0,
                'Average Loss': trades_df[trades_df['realized_pnl'] <= 0]['realized_pnl'].mean() if len(trades_df[trades_df['realized_pnl'] <= 0]) > 0 else 0,
                'Total Realized P&L': trades_df['realized_pnl'].sum(),
                'Average Return (%)': trades_df['return_pct'].mean(),
                'Average Hold Days': trades_df['hold_days'].mean(),
                'Best Trade': trades_df['realized_pnl'].max(),
                'Worst Trade': trades_df['realized_pnl'].min()
            }

            # Profit factor
            gross_profit = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl'].sum()
            gross_loss = abs(trades_df[trades_df['realized_pnl'] <= 0]['realized_pnl'].sum())
            stats['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else 0

            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            stats_df.to_excel(writer, sheet_name='Trade Statistics')
            logger.info("Exported trade statistics")

        # Sheet 5: Performance by Signal Type
        if trades_df is not None and len(trades_df) > 0:
            signal_perf = trades_df.groupby('signal').agg({
                'realized_pnl': ['count', 'sum', 'mean'],
                'return_pct': 'mean',
                'hold_days': 'mean'
            }).round(2)

            signal_perf.to_excel(writer, sheet_name='Performance by Signal')
            logger.info("Exported performance by signal type")

        # Sheet 6: Recent Open Positions (if any)
        position_files = sorted(data_dir.glob('open_positions_*.csv'))
        if position_files:
            latest_positions = pd.read_csv(position_files[-1])
            latest_positions.to_excel(writer, sheet_name='Current Positions', index=False)
            logger.info(f"Exported current positions: {len(latest_positions)} positions")

    logger.info(f"Excel report saved to: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    export_to_excel()
    print("Excel export complete!")
