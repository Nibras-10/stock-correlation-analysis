import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

def conditional_entropy(prob_matrix):
    """Compute conditional entropy H(Y|X)."""
    epsilon = 1e-10
    
    # Ensure no zeros in the matrix
    prob_matrix = prob_matrix + epsilon
    
    # Normalize to get joint probability
    joint_prob = prob_matrix / np.sum(prob_matrix)
    
    # Marginal probability of X
    prob_x = np.sum(joint_prob, axis=1, keepdims=True)
    
    # Conditional probability P(Y|X)
    cond_prob = joint_prob / prob_x
    
    # Calculate conditional entropy
    log_cond_prob = np.log2(cond_prob)
    cond_entropy = -np.sum(joint_prob * log_cond_prob)
    
    return cond_entropy

def transfer_entropy(a, b, bins=10):
    """Compute Transfer Entropy TE(A -> B)."""
    # Create the 2D histogram
    joint_hist_2d, x_edges, y_edges = np.histogram2d(a[:-1], b[:-1], bins=bins)
    
    # Create arrays for the three dimensions
    a_past = a[:-1]
    b_past = b[:-1]
    b_future = b[1:]
    
    # Manual 3D histogram calculation
    a_bins = np.linspace(min(a_past), max(a_past), bins+1)
    b_past_bins = np.linspace(min(b_past), max(b_past), bins+1)
    b_future_bins = np.linspace(min(b_future), max(b_future), bins+1)
    
    # Bin indices for each point
    a_indices = np.digitize(a_past, a_bins) - 1
    a_indices = np.clip(a_indices, 0, bins-1)  # Ensure within bounds
    
    b_past_indices = np.digitize(b_past, b_past_bins) - 1
    b_past_indices = np.clip(b_past_indices, 0, bins-1)
    
    b_future_indices = np.digitize(b_future, b_future_bins) - 1
    b_future_indices = np.clip(b_future_indices, 0, bins-1)
    
    # Fill the 3D histogram
    hist_3d = np.zeros((bins, bins, bins))
    for i in range(len(a_past)):
        hist_3d[a_indices[i], b_past_indices[i], b_future_indices[i]] += 1
    
    # Add small epsilon to avoid zeros
    epsilon = 1e-10
    joint_hist_2d = joint_hist_2d + epsilon
    hist_3d = hist_3d + epsilon
    
    # Calculate conditional entropies
    H_B_given_Bpast = conditional_entropy(np.sum(hist_3d, axis=0))
    H_B_given_Bpast_Anow = conditional_entropy(hist_3d)
    
    return H_B_given_Bpast - H_B_given_Bpast_Anow

def laggedCorr_TE(a, b, max_lag=20):
    """
    Compute Spearman Correlation and Transfer Entropy across different lags.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    
    # Handle potential issues with the data
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan, np.nan
        
    corrs = []
    lags = []
    transfer_entropies = []

    for lag in range(-max_lag, max_lag + 1):
        try:
            if lag < 0:
                # Ensure arrays are of equal length
                min_len = min(len(a[-lag:]), len(b[:lag]))
                if min_len > 10:  # Only calculate if we have enough data
                    a_slice = a[-lag:][:min_len]
                    b_slice = b[:lag][:min_len]
                    corr, _ = spearmanr(a_slice, b_slice)
                    te = transfer_entropy(a_slice, b_slice)
                else:
                    continue
            elif lag > 0:
                min_len = min(len(a[:-lag]), len(b[lag:]))
                if min_len > 10:
                    a_slice = a[:-lag][:min_len]
                    b_slice = b[lag:][:min_len]
                    corr, _ = spearmanr(a_slice, b_slice)
                    te = transfer_entropy(a_slice, b_slice)
                else:
                    continue
            else:
                corr, _ = spearmanr(a, b)
                te = transfer_entropy(a, b)
            
            # Skip if results are invalid
            if np.isnan(corr) or np.isnan(te) or np.isinf(corr) or np.isinf(te):
                continue
                
            corrs.append(corr)
            lags.append(lag)
            transfer_entropies.append(te)
        except Exception as e:
            print(f"Error at lag {lag}: {str(e)}")
            continue

    # Check if we have any valid results
    if not corrs:
        return np.nan, np.nan, np.nan
        
    sorted_corrs = sorted(zip(corrs, lags), key=lambda x: abs(x[0]), reverse=True)
    max_corr, best_lag = sorted_corrs[0]

    if best_lag == 0 and len(sorted_corrs) > 1 and abs(sorted_corrs[1][0]) >= 0.95 * abs(max_corr):
        max_corr, best_lag = sorted_corrs[1]

    return max_corr, best_lag, transfer_entropies[lags.index(best_lag)]

# Load stock data
data = pd.read_csv('stocks.csv', parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

tickers = data.columns.tolist()
results = []

# Process stock pairs with more detailed error reporting
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        try:
            stock_a = data.iloc[:, i]
            stock_b = data.iloc[:, j]
            print(f"\nProcessing {tickers[i]} vs {tickers[j]}...")
            
            # Check if data is valid
            print(f"Stock A ({tickers[i]}) stats: min={stock_a.min():.2f}, max={stock_a.max():.2f}")
            print(f"Stock B ({tickers[j]}) stats: min={stock_b.min():.2f}, max={stock_b.max():.2f}")
            
            # Check for constant values which would break correlation
            if stock_a.std() == 0 or stock_b.std() == 0:
                print(f"Warning: One of the stocks has zero standard deviation!")
                results.append((tickers[i], tickers[j], np.nan, np.nan, np.nan))
                continue
                
            corr, lag, te = laggedCorr_TE(stock_a.values, stock_b.values)
            results.append((tickers[i], tickers[j], corr, lag, te))
            print(f"Results: Correlation={corr:.4f}, Lag={lag}, TE={te:.6f}")
        except Exception as e:
            print(f"Error processing {tickers[i]} vs {tickers[j]}: {str(e)}")
            results.append((tickers[i], tickers[j], np.nan, np.nan, np.nan))

results_df = pd.DataFrame(results, columns=['Stock A', 'Stock B', 'Correlation', 'Lag', 'Transfer Entropy'])
results_df.to_csv('stock_correlation_results.csv', index=False)

print("\nAnalysis complete with Conditional Entropy-Based Transfer Entropy.")
