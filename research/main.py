import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import time
from requests.exceptions import ReadTimeout, ConnectionError
import strategies
import metrics
import regime_seperation


asset_list = [
    "SPY", "QQQ", "DIA",
    "EWJ", "EEM", "VGK",
    "TLT", "IEF", "BND",
    "GLD", "SLV", "USO",
    "EURUSD=X", "JPY=X", "AUDUSD=X",
    "BTC-USD", "ETH-USD"
]

rule_grid = {
    'ATR': {
        'atr_len' : [7, 14]
    },
    'SMA_long': {
        'fast' : [5, 10, 13, 20],
        'slow' : [21, 50, 100, 200]
    },
    'SMA_short': {
        'fast': [5, 10, 13, 20],
        'slow': [21, 50, 100, 200]
    },
    'SMA_combined': {
        'fast': [5, 10, 13, 20],
        'slow': [21, 50, 100, 200]
    },
    'BB_RSI': {
        'bb_window': [10, 20],
        'bb_std': [1.5, 2.0],
        'rsi_period': [7, 14],
        'rsi_oversold': [30, 35],
        'rsi_overbought': [65, 70]
    },
    'MACD': {
        'fast': [8, 12],
        'slow': [24, 26],
        'signal': [9, 18]
    },
    'SnR': {
        'window' : [5, 10, 15],
        'duration' : [10, 20, 30],
        'atr' : {
            'atr_len' : [7, 14]
        },
        'atr_buffer' : [0.1, 0.2]
    },
    'TSMOM': {
        'lookback': [21, 63, 126, 252],
    },
    'BoS': {
        'lookback': [10, 20],
        'window' : [5],
        'duration' : [10, 20, 30],
        'atr' : {
            'atr_len' : [7, 14]
        },
        'atr_buffer' : [0.1, 0.2]
    },
    # 10, 15
    'FVG': {
    }
}

def get_data(ticker, max_retries=3, initial_delay=2):
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, interval="1d", start="2000-01-01", end="2024-01-01", timeout=30)
            df.columns = df.columns.get_level_values(0)
            df = df[['High', 'Low', 'Open', 'Close']].dropna()
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            return df
        
        except (ReadTimeout, ConnectionError, TimeoutError) as e:
            delay = initial_delay * (2**attempt)
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise
        
        except Exception as e:
            raise

    raise Exception(f'failed to download {ticker}')
        

def walk_forward(df, strategy_fn, strategy_params=None,train_years=3, test_years=1):

    df = df.copy()
    df = df.sort_index() 
    results = []

    if strategy_params is None:
        strategy_params = {}

    start = df.index.min()

    while True:
        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        # break if test window exceeds data
        if test_end > df.index.max():
            break

        # slice windows
        train = df[(df.index >= train_start) & (df.index < train_end)].copy()
        test = df[(df.index >= train_end) & (df.index < test_end)].copy()

        # if too small skip
        if len(train) < 300:
            start = start + pd.DateOffset(years=test_years)
            continue

        # train + test combined for MA calculation
        combined = pd.concat([train, test]).sort_index()

        # apply strategy
        window_results = strategy_fn(combined, **strategy_params)

        # keep only test window results
        window_results = window_results.loc[test.index]

        results.append(window_results)

        # move forward
        start = start + pd.DateOffset(years=test_years)

    final = pd.concat(results)
    return final

def generate_param_combinations(param_dict):
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

def expand_params(param_dict):
    flat_params = [{}]

    for key, val in param_dict.items():
        if isinstance(val, dict):
            # Recursively expand nested dict
            sub_expanded = expand_params(val)
            new_flat = []
            for base in flat_params:
                for sub in sub_expanded:
                    new_flat.append({**base, key: sub})
            flat_params = new_flat

        elif isinstance(val, list):
            new_flat = []
            for base in flat_params:
                for item in val:
                    new_flat.append({**base, key: item})
            flat_params = new_flat

        else:
            raise ValueError("Parameter grid values must be dict or list")

    return flat_params

def realized_volatility_regimes(returns, window=20):

    # Calculate rolling volatility
    rolling_vol = returns.rolling(window).std()
    
    # Define high/low volatility threshold (median or 60th percentile)
    threshold = rolling_vol.quantile(0.6)
    
    # Create regime: 0 = low vol, 1 = high vol
    regime = (rolling_vol > threshold).astype(int)
    
    return regime

def safe_granger_test(result):
    try:
        # Check if there's sufficient variation
        if result['signal'].std() == 0:
            return None, None  # No signal variation
        if result['ret'].std() == 0:
            return None, None  # No return variation
        
        # Check if there are enough non-zero signals
        if (result['signal'] != 0).sum() < 10:
            return None, None  # Too few signals
        
        g1, g2 = metrics.granger_test(result)
        return g1, g2
    
    except Exception as e:
        print(f"    Granger test failed: {str(e)}")
        return None, None

strategy_map = {
    'SMA_long' : strategies.sma_long_strategy, 
    'SMA_short' : strategies.sma_short_strategy, 
    'SMA_combined' : strategies.sma_combined_strategy, 
    'BB_RSI' : strategies.bbnrsi_strategy, 
    'SnR' : strategies.snr_strategy, 
    'FVG' : strategies.fvg_strategy,
    'BoS' : strategies.bos_strategy, 
    'TSMOM' : strategies.tsmom_strategy 
}

strategy_param_space = {}

for strat_name, param_dict in rule_grid.items():
    strategy_param_space[strat_name] = expand_params(param_dict)

results = []
models = {}

for ticker in asset_list:
    df = get_data(ticker)
    
    market_ret = df['Close'].pct_change().dropna()
    
    try:
        model = MarkovRegression(
            market_ret,
            k_regimes=2,
            trend="c",
            switching_variance=True
        )

        models[ticker] = model.fit(
            maxiter=1000,
            method='bfgs',
            disp=False,
            search_reps=20)
        
    except (np.linalg.LinAlgError, ValueError, Exception) as e:
        try:
            model = MarkovRegression(
                market_ret,
                k_regimes=2,
                trend="c",
                switching_variance=False  # Fallback
            )
            
            models[ticker] = model.fit(maxiter=500, method='bfgs', disp=False)

        except Exception as e2:
            models[ticker] = None

    for strat_name, strat_fn in strategy_map.items():
        param_list = strategy_param_space.get(strat_name, [])

        print(f"  Strategy: {strat_name} | {len(param_list)} parameter sets")

        for params in param_list:

            # run walk-forward
            result = walk_forward(df, strat_fn, strategy_params=params)
            # result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
            perf = metrics.compute_performance_metrics(result)
            jensens_alpha = metrics.compute_jensens_alpha(result)
            statistical_tests_results = metrics.statistical_tests(result)
            g1, g2 = safe_granger_test(result)

            try: 
                real_perf, mean_random_perf, permutation_pvalue = metrics.random_permutation_test(result)
            except Exception as e:
                real_perf, mean_random_perf, permutation_pvalue = None, None, None

            try:
                nwa_tvalue, nwa_pvalue = metrics.newey_west_alpha_test(result)
            except Exception as e:
                nwa_tvalue, nwa_pvalue = None, None

            has_variation = result['strategy_ret'].std() > 1e-10

            try:
                if has_variation:
                    test_results = metrics.stationarity_tests(result['strategy_ret'])
                    adf_p = test_results['adf_p']
                    kpss_p = test_results['kpss_p']
                else:
                    adf_p, kpss_p = None, None
            except Exception as e:
                adf_p, kpss_p = None, None

            try:
                if has_variation:
                    s1, s2, diff, z, lw_sharpe_pvalue = metrics.ledoit_wolf_sharpe_test(result['strategy_ret'], result['ret'])
                else:
                    s1, s2, diff, z, lw_sharpe_pvalue = None, None, None, None, None
            except Exception as e:
                s1, s2, diff, z, lw_sharpe_pvalue = None, None, None, None, None

            try:
                if has_variation:
                    bootstrap_mean, mean2, bootstrap_pvalue = metrics.bootstrap_significance(result['strategy_ret'])
                else:
                    bootstrap_mean, mean2, bootstrap_pvalue = None, None, None
            except Exception as e:
                bootstrap_mean, mean2, bootstrap_pvalue = None, None, None

            try: 
                innov_test = metrics.innovation_test(result['strategy_ret'], result['signal'])
            except Exception as e:
                innov_test = None

            try: 
                factor_reg = metrics.factor_regression(result)
                if factor_reg is None:
                    print(f"  ⚠ Factor regression returned None for {ticker} - {strat_name}")
            except Exception as e:
                print(f"  ✗ Factor regression failed for {ticker} - {strat_name}: {str(e)}")
                factor_reg = None

            if models[ticker] is None:
                regime = realized_volatility_regimes(market_ret)
            else:
                regime = models[ticker].smoothed_marginal_probabilities.idxmax(axis=1)

            low_vol, high_vol = regime_seperation.seperate_by_regimes(result, regime)
            high_low_vol_analyze = regime_seperation.analyze_by_regime(low_vol, high_vol)

            fred_series = regime_seperation.get_daily_fred_series('USREC', result.index)
            recession_dates,  non_recession_dates = regime_seperation.seperate_by_recession(result, fred_series)
            recession_nonrecession_analyze = regime_seperation.analyze_by_recession(recession_dates, non_recession_dates)

            results.append({
                'asset' : ticker,
                'strategy' : strat_name,
                'params' : params,
                'perf_metrics' : perf,
                'jensens_alpha' : jensens_alpha,
                'statistical_tests' : statistical_tests_results,
                'Does signal cause returns?' : g1,
                'Does return cause signal? (control test)' : g2,
                'permutation_pvalue' : permutation_pvalue,
                'newey west alpha t value' : nwa_tvalue,
                'newey west alpha p value' : nwa_pvalue,
                'adf_p' : adf_p,
                'kpss_p' : kpss_p,
                'ledoit wolf sharpe p value' : lw_sharpe_pvalue,
                'bootstrap significance' : bootstrap_pvalue,
                'high n low volatility performance metrics' : high_low_vol_analyze,
                'recession n nonrecession performance metrics' : recession_nonrecession_analyze,
                'innovation test beta' : innov_test['beta'] if innov_test else None,
                'innovation test pvalue' : innov_test['p_value'] if innov_test else None,
                'innovation test stat' : innov_test['t_stat'] if innov_test else None,
                'factor regression alpha' : factor_reg['alpha'] if factor_reg else None,
                'factor regression alpha pvalue' : factor_reg['alpha_p'] if factor_reg else None,
                'factor regression alpha tvalue' : factor_reg['alpha_t'] if factor_reg else None,
                'factor regression beta' : factor_reg['betas'] if factor_reg else None,
            })

def results_to_excel_by_strategy(results, filename='backtest_by_strategy_{strategy_name}.xlsx'):
    
    # Group results by asset and strategy
    grouped = {}
    for result in results:
        key = (result['asset'], result['strategy'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        for (asset, strategy), group_results in grouped.items():
            # Create sheet name (limit to 31 characters)
            sheet_name = f"{asset}_{strategy}"[:31]
            
            # Create rows for this sheet
            rows = []
            
            for result in group_results:
                row = {
                    'Asset': result['asset'],
                    'Strategy': result['strategy'],
                }
                
                # Flatten and add parameters
                params = result['params']
                def flatten_params(d, parent_key=''):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}_{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_params(v, new_key).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flat_params = flatten_params(params)
                row.update(flat_params)
                
                perf_metrics = result['perf_metrics']
                if isinstance(perf_metrics, dict):
                    row.update(perf_metrics)
                elif isinstance(perf_metrics, pd.Series):
                    row.update(perf_metrics.to_dict())
                elif isinstance(perf_metrics, (list, tuple)):
                    for i, val in enumerate(perf_metrics):
                        row[f'perf_metric_{i}'] = val
                else:
                    row['perf_metrics'] = str(perf_metrics)
                
                # Add other metrics
                row['jensens_alpha'] = result['jensens_alpha']
                row.update(result['statistical_tests'])
                row['signal_causes_returns'] = result['Does signal cause returns?']
                row['returns_cause_signal'] = result['Does return cause signal? (control test)']
                row['permutation_pvalue'] = result['permutation_pvalue']
                row['nw_alpha_tvalue'] = result['newey west alpha t value']
                row['nw_alpha_pvalue'] = result['newey west alpha p value']
                row['adf_p'] = result['adf_p']
                row['kpss_p'] = result['kpss_p']
                row['lw_sharpe_pvalue'] = result['ledoit wolf sharpe p value']
                row['bootstrap_pvalue'] = result['bootstrap significance']
                
                vol_metrics = result['high n low volatility performance metrics']
                
                # Add high volatility metrics
                for key, value in vol_metrics.items():
                    if 'high vol' in key:
                        new_key = f"hvol_{key.replace('in high vol', '').strip()}"
                        row[new_key] = value
                
                # Add low volatility metrics
                for key, value in vol_metrics.items():
                    if 'low vol' in key:
                        new_key = f"lvol_{key.replace('in low vol', '').strip()}"
                        row[new_key] = value
                
                rec_metrics = result['recession n nonrecession performance metrics']
                
                # Add recession metrics
                for key, value in rec_metrics.items():
                    if 'recession' in key and 'nonrecession' not in key:
                        new_key = f"rec_{key.replace('in recession', '').strip()}"
                        row[new_key] = value
                
                # Add non-recession metrics
                for key, value in rec_metrics.items():
                    if 'nonrecession' in key:
                        new_key = f"nonrec_{key.replace('in nonrecession', '').strip()}"
                        row[new_key] = value

                row['innovation_test_beta'] = result['innovation test beta']
                row['innovation_test_pvalue'] = result['innovation test pvalue']
                row['innovation_test_stat'] = result['innovation test stat']
                row['factor_regression_alpha'] = result['factor regression alpha']
                row['factor_regression_alpha_pvalue'] = result['factor regression alpha pvalue']
                row['factor_regression_alpha_tvalue'] = result['factor regression alpha tvalue']
                row['factor_regression_beta'] = result['factor regression beta']
                
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = 100
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✓ Results saved to {filename}")
    print(f"✓ Total sheets created: {len(grouped)}")

# def results_to_excel_by_strategy(results, filename='backtest_by_strategy_fvg.xlsx'):
    
#     # Group results by asset and strategy
#     grouped = {}
#     for result in results:
#         key = (result['asset'], result['strategy'])
#         if key not in grouped:
#             grouped[key] = []
#         grouped[key].append(result)
    
#     # Check if we have any results
#     if not grouped:
#         print("No results to write!")
#         return
    
#     with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
        
#         for (asset, strategy), group_results in grouped.items():
#             # Create sheet name (limit to 31 characters)
#             sheet_name = f"{asset}_{strategy}"[:31]
            
#             # Create rows for this sheet
#             rows = []
            
#             for result in group_results:
#                 row = {
#                     'Asset': result['asset'],
#                     'Strategy': result['strategy'],
#                 }
                
#                 # [Rest of your row creation code remains the same]
#                 params = result['params']
#                 def flatten_params(d, parent_key=''):
#                     items = []
#                     for k, v in d.items():
#                         new_key = f"{parent_key}_{k}" if parent_key else k
#                         if isinstance(v, dict):
#                             items.extend(flatten_params(v, new_key).items())
#                         else:
#                             items.append((new_key, v))
#                     return dict(items)
                
#                 flat_params = flatten_params(params)
#                 row.update(flat_params)
                
#                 perf_metrics = result['perf_metrics']
#                 if isinstance(perf_metrics, dict):
#                     row.update(perf_metrics)
#                 elif isinstance(perf_metrics, pd.Series):
#                     row.update(perf_metrics.to_dict())
#                 elif isinstance(perf_metrics, (list, tuple)):
#                     for i, val in enumerate(perf_metrics):
#                         row[f'perf_metric_{i}'] = val
#                 else:
#                     row['perf_metrics'] = str(perf_metrics)
                
#                 # Add other metrics
#                 row['jensens_alpha'] = result['jensens_alpha']
#                 row.update(result['statistical_tests'])
#                 row['signal_causes_returns'] = result['Does signal cause returns?']
#                 row['returns_cause_signal'] = result['Does return cause signal? (control test)']
#                 row['permutation_pvalue'] = result['permutation_pvalue']
#                 row['nw_alpha_tvalue'] = result['newey west alpha t value']
#                 row['nw_alpha_pvalue'] = result['newey west alpha p value']
#                 row['adf_p'] = result['adf_p']
#                 row['kpss_p'] = result['kpss_p']
#                 row['lw_sharpe_pvalue'] = result['ledoit wolf sharpe p value']
#                 row['bootstrap_pvalue'] = result['bootstrap significance']
                
#                 vol_metrics = result['high n low volatility performance metrics']
                
#                 for key, value in vol_metrics.items():
#                     if 'high vol' in key:
#                         new_key = f"hvol_{key.replace('in high vol', '').strip()}"
#                         row[new_key] = value
                
#                 for key, value in vol_metrics.items():
#                     if 'low vol' in key:
#                         new_key = f"lvol_{key.replace('in low vol', '').strip()}"
#                         row[new_key] = value
                
#                 rec_metrics = result['recession n nonrecession performance metrics']
                
#                 for key, value in rec_metrics.items():
#                     if 'recession' in key and 'nonrecession' not in key:
#                         new_key = f"rec_{key.replace('in recession', '').strip()}"
#                         row[new_key] = value
                
#                 for key, value in rec_metrics.items():
#                     if 'nonrecession' in key:
#                         new_key = f"nonrec_{key.replace('in nonrecession', '').strip()}"
#                         row[new_key] = value

#                 row['innovation_test_beta'] = result['innovation test beta']
#                 row['innovation_test_pvalue'] = result['innovation test pvalue']
#                 row['innovation_test_stat'] = result['innovation test stat']
#                 row['factor_regression_alpha'] = result['factor regression alpha']
#                 row['factor_regression_alpha_pvalue'] = result['factor regression alpha pvalue']
#                 row['factor_regression_alpha_tvalue'] = result['factor regression alpha tvalue']
#                 row['factor_regression_beta'] = result['factor regression beta']
                
#                 rows.append(row)
            
#             # Create DataFrame and write
#             df = pd.DataFrame(rows)
            
#             if df.empty:
#                 print(f"Warning: Empty DataFrame for {sheet_name}")
#                 continue
                
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
    
#     print(f"✓ Results saved to {filename}")
#     print(f"✓ Total sheets created: {len(grouped)}")

results_to_excel_by_strategy(results, 'backtest_by_strategy_{strategy_name}.xlsx')