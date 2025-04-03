import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def laggedCorr(a, b):
    """
    Calculate the maximum correlation between two series considering different lags.
    Using Spearman rank correlation instead of Pearson.
    Returns the highest correlation value and its corresponding lag.
    Implements improvements to reduce spurious 0-lag results.
    """
    a = np.array(a)
    b = np.array(b)
    
    # Reduce maximum lag to a more reasonable value based on data size
    max_lag = min(60, len(a) // 10)  # Using 10% of data length as maximum lag
    
    correlations = []
    lags = []
    p_values = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Align series with negative lag
            series_a = a[-lag:]
            series_b = b[:lag]
        elif lag > 0:
            # Align series with positive lag
            series_a = a[:-lag]
            series_b = b[lag:]
        else:
            # No lag
            series_a = a
            series_b = b
        
        # Only calculate if there's enough data
        if len(series_a) > 10 and len(series_b) > 10:
            # Get both correlation and p-value
            corr_result = stats.spearmanr(series_a, series_b)
            corr = corr_result[0]
            p_val = corr_result[1]
            
            correlations.append(corr)
            lags.append(lag)
            p_values.append(p_val)
        else:
            # Not enough data for reliable correlation
            correlations.append(0)
            lags.append(lag)
            p_values.append(1.0)
    
    # Filter for statistical significance
    significant_indices = [i for i, p in enumerate(p_values) if p < 0.05]
    
    if not significant_indices:
        return 0, 0  # No significant correlation found
    
    # Among significant correlations, find the one with highest absolute value
    abs_corrs = [abs(correlations[i]) for i in significant_indices]
    max_abs_corr_idx = significant_indices[abs_corrs.index(max(abs_corrs))]
    
    # Use a bias against selecting lag 0 to overcome the natural preference for lag 0
    # Only select lag 0 if it's significantly better than other lags
    if lags[max_abs_corr_idx] == 0:
        # Find the next best lag that's not 0
        non_zero_indices = [i for i in significant_indices if lags[i] != 0]
        if non_zero_indices:
            next_best_corr = max([abs(correlations[i]) for i in non_zero_indices])
            if abs(correlations[max_abs_corr_idx]) - next_best_corr < 0.05:
                # If lag 0 isn't substantially better, choose non-zero lag
                non_zero_abs_corrs = [abs(correlations[i]) for i in non_zero_indices]
                next_best_idx = non_zero_indices[non_zero_abs_corrs.index(next_best_corr)]
                return correlations[next_best_idx], lags[next_best_idx]
    
    return correlations[max_abs_corr_idx], lags[max_abs_corr_idx]

# Read the stock data from CSV file
data = pd.read_csv('stocks.csv', parse_dates=['Date'])

data.set_index('Date', inplace=True)

data.ffill(inplace=True)
data.bfill(inplace=True)

print("Remaining NaN values:", data.isnull().sum().sum())

tickers = data.columns.tolist()
results = []

# Calculate lagged correlations for all pairs of stocks
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        stock_a = data.iloc[:, i]
        stock_b = data.iloc[:, j]
        corr, lag = laggedCorr(stock_a.values, stock_b.values)
        results.append((tickers[i], tickers[j], corr, lag))

# Save categorized correlation results
categorized_results = pd.DataFrame(results, columns=['Stock A', 'Stock B', 'Correlation', 'Lag'])
categorized_results['Category'] = categorized_results['Correlation'].apply(
    lambda x: 'Positive' if x > 0.5 else ('Negative' if x < -0.5 else 'No Correlation')
)
categorized_results.to_csv('categorized_correlations.csv', index=False)

# Create a DataFrame for correlation matrix
correlation_matrix = pd.DataFrame(index=tickers, columns=tickers)
for stock_a, stock_b, corr, _ in results:
    correlation_matrix.loc[stock_a, stock_b] = corr
    correlation_matrix.loc[stock_b, stock_a] = corr
correlation_matrix = correlation_matrix.astype(float)
correlation_matrix.to_csv('stock_correlation_matrix.csv')

# Create a heatmap
mask = np.eye(len(tickers), dtype=bool)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            xticklabels=True, yticklabels=True, mask=mask)
plt.title('Spearman Correlation Heatmap between NSE Stocks (Self-Correlations Removed)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# Find the top correlations (excluding self-correlations)
top_correlations = sorted(results, key=lambda x: abs(x[2]), reverse=True)[:20]

# Save the top correlations with lag values
top_corr_df = pd.DataFrame(top_correlations, columns=['Stock A', 'Stock B', 'Correlation', 'Lag'])
top_corr_df.to_csv('top_correlations.csv', index=False)

print("\nAnalysis complete. Output files saved:")
print("1. stock_correlation_matrix.csv - Full Spearman correlation matrix (with NaN diagonals)")
print("2. categorized_correlations.csv - Categorized correlations (Positive, Negative, No Correlation)")
print("3. correlation_heatmap.png - Heatmap visualization of Spearman correlations (self-correlations removed)")
print("4. top_correlations.csv - Top 20 stock correlations with lag values")