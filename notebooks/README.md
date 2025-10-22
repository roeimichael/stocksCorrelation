# Data Analysis Notebook

This folder contains Jupyter notebooks for analyzing the S&P 500 analog pattern matching dataset.

## Available Notebooks

### data_analysis.ipynb

Comprehensive macro-level analysis of the dataset including:

**12 Analysis Sections:**
1. **Load Data** - Import all dataset files
2. **Dataset Overview** - High-level statistics
3. **Ticker Distribution** - Analysis of included stocks
4. **Price Statistics** - Price ranges and distributions
5. **Returns Distribution** - Statistical analysis of daily returns
6. **Volatility Analysis** - Most and least volatile stocks
7. **Performance Analysis** - Best/worst performers vs S&P 500
8. **Window Label Distribution** - Up/down day balance
9. **Windows per Stock** - Data completeness check
10. **Time Series Visualization** - Price evolution charts
11. **Summary Statistics** - Comprehensive metrics table
12. **Data Quality Checks** - Missing data and extreme values

## How to Run

### Option 1: Jupyter Notebook

```bash
# Install dependencies (if not already installed)
pip install -r ../requirements.txt

# Start Jupyter
jupyter notebook

# Open data_analysis.ipynb in the browser
```

### Option 2: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab

# Navigate to notebooks/data_analysis.ipynb
```

### Option 3: VS Code

1. Open the project in VS Code
2. Install the "Jupyter" extension
3. Open `notebooks/data_analysis.ipynb`
4. Select the Python kernel from your `.venv`
5. Run cells interactively

## Prerequisites

Make sure you have run the preprocessing script first:

```bash
python scripts/preprocess.py
```

This creates the required data files:
- `data/raw/adj_close_prices.parquet`
- `data/processed/returns.parquet`
- `data/processed/windows.pkl`

## Output

The notebook generates:
- **Statistics tables** - Key metrics about the dataset
- **Visualizations** - Histograms, scatter plots, time series, pie charts
- **Quality checks** - Data completeness and integrity analysis

All visualizations are rendered inline in the notebook.

## Dataset Summary

- **502 stocks** (S&P 500 + index)
- **500 trading days** (2 years)
- **125,500 pattern windows**
- Date range: 2023-10-23 to 2025-10-21
