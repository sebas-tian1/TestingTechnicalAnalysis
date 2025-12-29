import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
import pandas_datareader.data as web
from datetime import datetime
import io
import requests
import zipfile
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import norm
from scipy.stats import ttest_rel, wilcoxon, binomtest
from statsmodels.tsa.stattools import grangercausalitytests
from fredapi import Fred

fred_key ='your_fred_api_key_here'  # Replace with your actual FRED API key

fred = Fred(api_key=fred_key)

RISK_FREE_RATE = fred.get_series(series_id='TB3MS', observation_start = '2000-01-01', observation_end='2024-01-01').iloc[-1]

def compute_performance_metrics(df, risk_free_rate=RISK_FREE_RATE):
    df = df.copy()
    
    # Make sure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    init_equity = 10000
    
    # Correct equity curve calculation
    df['equity'] = init_equity * (1 + df['strategy_ret']).cumprod()
    
    # 1. CAGR - Use actual date range from the results
    total_years = (df.index[-1] - df.index[0]).days / 365.25
    final_equity = df['equity'].iloc[-1]
    cagr = (final_equity / init_equity) ** (1 / total_years) - 1
    
    # 2. Max Drawdown
    roll_max = df['equity'].cummax()
    drawdown = (df['equity'] - roll_max) / roll_max
    max_drawdown = drawdown.min()
    
    # 3. Sharpe Ratio (annualized)
    daily_rf = risk_free_rate / 252
    excess_return = df['strategy_ret'] - daily_rf
    sharpe = np.sqrt(252) * excess_return.mean() / excess_return.std() if excess_return.std() != 0 else np.nan
    
    # 4. Sortino Ratio (only downside volatility)
    downside = excess_return[excess_return < 0]
    sortino = np.sqrt(252) * excess_return.mean() / downside.std() if len(downside) > 0 and downside.std() != 0 else np.nan
    
    # 5. Profit Factor
    gross_profit = df['strategy_ret'][df['strategy_ret'] > 0].sum()
    gross_loss = abs(df['strategy_ret'][df['strategy_ret'] < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
    
    # 6. Rolling Sharpe (30-day window)
    rolling_excess = df['strategy_ret'].rolling(30).mean() - (risk_free_rate / 252)
    df['Rolling_Sharpe'] = (rolling_excess / df['strategy_ret'].rolling(30).std()) * np.sqrt(252)
    
    metrics = {
        'CAGR': cagr,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Profit Factor': profit_factor,
        '30 days Rolling Sharpe': df['Rolling_Sharpe'].iloc[-30:].mean()
    }
    
    return metrics, df

def compute_jensens_alpha(df, risk_free_rate=RISK_FREE_RATE):
    rp = (1 + df['strategy_ret'].mean())**252 - 1
    rm = (1 + df['ret'].mean())**252 - 1
    cov = np.cov(df['strategy_ret'].dropna(), df['ret'].dropna())[0,1]
    var = np.var(df['ret'].dropna())
    beta = cov / var
    alpha = rp - (risk_free_rate + beta * (rm - risk_free_rate))

    return alpha

def statistical_tests(res):
    strat = res['strategy_ret']
    bench = res['ret']
    diff = strat - bench
    
    tests = {}

    # Paired t-test
    t_stat, p_t = ttest_rel(strat, bench)
    tests['paired_t_p'] = p_t

    # Wilcoxon test (non-parametric)
    w_stat, p_w = wilcoxon(strat, bench)
    tests['wilcoxon_p'] = p_w

    # Directional accuracy
    future = res['ret'].shift(-1)
    correct = ((res['signal']==1)&(future>0)) | ((res['signal']==0)&(future<0))
    accuracy = correct.mean()
    tests['accuracy'] = accuracy

    # Binomial test for accuracy vs 0.5
    tests['accuracy_binom_p'] = binomtest(
        correct.sum(), n=len(correct), p=0.5
    )

    return tests

def granger_test(res):
    df = res[['ret', 'signal']].dropna()
    # Does signal cause returns?
    g1 = grangercausalitytests(df[['ret','signal']], maxlag=5, verbose=False)
    # Does return cause signal? (control test)
    g2 = grangercausalitytests(df[['signal','ret']], maxlag=5, verbose=False)
    return g1, g2

def random_permutation_test(res, n=5000):
    real_perf = (1 + res['strategy_ret']).prod()

    random_perf = []
    for _ in range(n):
        shuffled = res['ret'].sample(frac=1, replace=False).values
        shuffled_strat = res['signal'].values * shuffled
        random_perf.append((1 + shuffled_strat).prod())

    random_perf = np.array(random_perf)
    p_value = np.mean(random_perf > real_perf)

    return real_perf, random_perf.mean(), p_value

def newey_west_alpha_test(res):
    y = res['strategy_ret'] - res['ret']      # strategy excess over market
    X = np.ones(len(y))                       # constant only
    model = sm.OLS(y, X)
    nw = model.fit(cov_type='HAC', cov_kwds={'maxlags':5})
    return nw.tvalues[0], nw.pvalues[0]

def stationarity_tests(series):
    return {
        'adf_p': adfuller(series.dropna())[1],
        'kpss_p': kpss(series.dropna(), nlags='auto')[1]
    }

def ledoit_wolf_sharpe_test(r1, r2):
    n = len(r1)
    mean1, mean2 = np.mean(r1), np.mean(r2)
    std1, std2  = np.std(r1, ddof=1), np.std(r2, ddof=1)

    sharpe1, sharpe2 = mean1/std1, mean2/std2
    diff = sharpe1 - sharpe2

    # LW correction components
    term1 = (mean1**2 / (2 * std1**4)) * np.var((r1 - mean1)**2)
    term2 = (mean2**2 / (2 * std2**4)) * np.var((r2 - mean2)**2)
    var_diff = term1 + term2

    z = diff / np.sqrt(var_diff / n)
    p = 1 - norm.cdf(z)  # one-sided test: is Sharpe1 > Sharpe2?

    return sharpe1, sharpe2, diff, z, p

def bootstrap_significance(strategy_returns, n_boot=5000):
    actual_mean = np.mean(strategy_returns)

    boot_means = []
    for _ in range(n_boot):
        random_sign = np.random.choice([-1, 1], size=len(strategy_returns))
        boot_means.append(np.mean(strategy_returns * random_sign))

    p_value = np.mean(np.array(boot_means) >= actual_mean)
    return actual_mean, np.mean(boot_means), p_value

def innovation_test(market, signal):
    arima = sm.tsa.ARIMA(market, order=(1,0,0)).fit()
    mean_resid = arima.resid 

    garch = arch_model(mean_resid, vol='Garch', p=1, q=1).fit(update_freq=0, disp='off')
    innov = garch.resid / garch.conditional_volatility

    test_df = pd.DataFrame({
    'innov': innov,
    'signal': signal
    }).dropna()

    X = sm.add_constant(test_df['signal'])
    model = sm.OLS(test_df['innov'], X).fit()
    return {
        'beta': model.params['signal'],
        'p_value': model.pvalues['signal'],
        't_stat': model.tvalues['signal'],
        'r2': model.rsquared,
        'n_obs': model.nobs
    }

def load_ff_data(url):
    """Load Fama-French factor data from URL"""
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    file = z.namelist()[0]
    
    # Read the entire file first
    with z.open(file) as f:
        lines = [line.decode('utf-8').strip() for line in f.readlines()]
    
    # Find where the actual data starts (after header rows)
    data_start = 0
    for i, line in enumerate(lines):
        # Look for a line that starts with a date (8 digits)
        if line and line.split(',')[0].strip().isdigit() and len(line.split(',')[0].strip()) == 8:
            data_start = i
            break
    
    # Get column names from the line before data starts
    header_line = lines[data_start - 1]
    columns = [col.strip() for col in header_line.split(',')]
    
    # Parse data rows
    data_rows = []
    for line in lines[data_start:]:
        if not line or line.startswith(' ') or 'Copyright' in line:
            break  # Stop at empty line or copyright notice
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == len(columns) and parts[0].isdigit() and len(parts[0]) == 8:
            data_rows.append(parts)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    
    # Rename first column to Date
    df = df.rename(columns={df.columns[0]: 'Date'})
    
    # Convert date with error handling
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
    
    # Remove any rows where date conversion failed
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with all NaN values
    df = df.dropna(how='all')
    
    return df

ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_Daily_CSV.zip"
mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_Daily_CSV.zip"

def factor_regression(res):
    try:
        # Load factor data
        factors_raw = load_ff_data(ff5_url)
        momentum = load_ff_data(mom_url)
        
        # Combine factors
        factors = factors_raw.join(momentum, how='inner')
        
        # Convert from percentage to decimal
        factors = factors / 100
        
        # Ensure res has datetime index
        if not isinstance(res.index, pd.DatetimeIndex):
            res.index = pd.to_datetime(res.index)
        
        # Normalize both indices (remove time component)
        res.index = res.index.normalize()
        factors.index = factors.index.normalize()
        
        # Merge strategy returns with factors
        merged = res[['strategy_ret']].join(factors, how='inner').dropna()
        
        if len(merged) < 30:  # Need minimum observations
            print(f"Warning: Only {len(merged)} observations after merging. Need at least 30.")
            return None
        
        # Prepare regression
        y = merged['strategy_ret']
        X = sm.add_constant(merged.drop(columns=['strategy_ret']))
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        return {
            "alpha": model.params['const'],
            "alpha_t": model.tvalues['const'],
            "alpha_p": model.pvalues['const'],
            "betas": model.params.drop('const').to_dict(),
            "beta_p_values": model.pvalues.drop('const').to_dict(),
            "beta_t_values": model.tvalues.drop('const').to_dict(),
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "n_obs": int(model.nobs),
            "start_date": merged.index.min().strftime('%Y-%m-%d'),
            "end_date": merged.index.max().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"Error in factor_regression: {str(e)}")
        import traceback
        traceback.print_exc()
        return None