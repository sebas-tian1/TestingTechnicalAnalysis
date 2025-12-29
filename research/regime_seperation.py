import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

fred_key ='your_fred_api_key_here'

fred = Fred(api_key=fred_key)

def seperate_by_regimes(results, regimes):
    common_dates = results.index.intersection(regimes.index)
    
    if len(common_dates) == 0:
        print("WARNING: No overlapping dates between results and regimes!")
        return results.copy(), pd.DataFrame()
    
    # Filter regimes untuk tanggal yang ada di results
    aligned_regimes = regimes.loc[common_dates]
    
    # Separate
    low_vol_dates = aligned_regimes[aligned_regimes == 0].index
    high_vol_dates = aligned_regimes[aligned_regimes == 1].index
    
    low_vol = results.loc[low_vol_dates].copy()
    high_vol = results.loc[high_vol_dates].copy()
    
    print(f"\nFound {len(low_vol)} low volatility days")
    print(f"Found {len(high_vol)} high volatility days")
    print(f"Total: {len(low_vol) + len(high_vol)} days")
    
    return low_vol, high_vol

def analyze_by_regime(low_vol, high_vol):
    # Low Vol Stats
    if len(low_vol) > 0:
        low_strat_ret = low_vol['strategy_ret'].mean() * 252 * 100
        low_strat_vol = low_vol['strategy_ret'].std() * np.sqrt(252) * 100
        low_strat_sharpe = low_vol['strategy_ret'].mean() / low_vol['strategy_ret'].std() * np.sqrt(252) if low_vol['strategy_ret'].std() > 0 else 0
        low_bh_ret = low_vol['ret'].mean() * 252 * 100
        low_bh_vol = low_vol['ret'].std() * np.sqrt(252) * 100
        low_bh_sharpe = low_vol['ret'].mean() / low_vol['ret'].std() * np.sqrt(252) if low_vol['ret'].std() > 0 else 0
        low_win = (low_vol['strategy_ret'] > 0).sum() / len(low_vol) * 100
        low_strat_total = (1 + low_vol['strategy_ret']).prod() - 1
        low_bh_total = (1 + low_vol['ret']).prod() - 1
    else:
        low_strat_ret = low_strat_vol = low_strat_sharpe = 0
        low_bh_ret = low_bh_vol = low_bh_sharpe = low_win = 0
        low_strat_total = low_bh_total = 0
    
    # High Vol Stats
    if len(high_vol) > 0:
        high_strat_ret = high_vol['strategy_ret'].mean() * 252 * 100
        high_strat_vol = high_vol['strategy_ret'].std() * np.sqrt(252) * 100
        high_strat_sharpe = high_vol['strategy_ret'].mean() / high_vol['strategy_ret'].std() * np.sqrt(252) if high_vol['strategy_ret'].std() > 0 else 0
        high_bh_ret = high_vol['ret'].mean() * 252 * 100
        high_bh_vol = high_vol['ret'].std() * np.sqrt(252) * 100
        high_bh_sharpe = high_vol['ret'].mean() / high_vol['ret'].std() * np.sqrt(252) if high_vol['ret'].std() > 0 else 0
        high_win = (high_vol['strategy_ret'] > 0).sum() / len(high_vol) * 100
        high_strat_total = (1 + high_vol['strategy_ret']).prod() - 1
        high_bh_total = (1 + high_vol['ret']).prod() - 1
    else:
        high_strat_ret = high_strat_vol = high_strat_sharpe = 0
        high_bh_ret = high_bh_vol = high_bh_sharpe = high_win = 0
        high_strat_total = high_bh_total = 0
    
    return {
        'strategy total return in low vol' : low_strat_total,
        'strategy total return in high vol' : high_strat_total,

        'buy n hold total return in low vol' : low_bh_total,
        'buy n hold total return in high vol' : high_bh_total,

        'ann. strategy return in low vol' : low_strat_ret,
        'ann. strategy return in high vol' : high_strat_ret,

        'ann. buy n hold return in low vol' : low_bh_ret,
        'ann. buy n hold return in high vol' : high_bh_ret,

        'ann. strategy volatility in low vol' : low_strat_vol,
        'ann. strategy volatility in high vol' : high_strat_vol,

        'ann. buy n hold volatility in low vol' : low_bh_vol,
        'ann. buy n hold volatility in high vol' : high_bh_vol,

        'strategy sharpe ratio in low vol' : low_strat_sharpe,
        'strategy sharpe ratio in high vol' : high_strat_sharpe,

        'buy n hold sharpe ratio in low vol' : low_strat_sharpe,
        'buy n hold sharpe ratio in high vol' : high_strat_sharpe,

        'strategy winrate in low vol' : low_win,
        'strategy winrate in high vol' : high_win
    }

def get_daily_fred_series(series_id, results_index, start='2000-01-01', end='2024-01-01', plot=False):

    # Download series
    series = fred.get_series(series_id=series_id, observation_start=start, observation_end=end)

    # Optional plot
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(series)
        plt.title(f"FRED Series: {series_id}")
        plt.grid(True)
        plt.show()

    # Convert to DataFrame
    df = series.to_frame(name=series_id)
    df.index = pd.to_datetime(df.index)

    # Resample to daily and forward-fill
    df_daily = df.resample("D").ffill()
    df_aligned = df_daily.reindex(results_index).ffill()

    return df_aligned

def seperate_by_recession(result, recession):
    
    if isinstance(recession, pd.DataFrame):
        recession = recession.iloc[:, 0]

    result.index = pd.to_datetime(result.index)
    recession.index = pd.to_datetime(recession.index)

    common_dates = result.index.intersection(recession.index)
    
    if len(common_dates) == 0:
        print("WARNING: No overlapping dates between results and recession!")
        return result.copy(), pd.DataFrame()
    
    # Filter regimes untuk tanggal yang ada di results
    recession_regimes = recession.loc[common_dates]
    
    # Separate
    non_recession_dates_idx = recession_regimes[recession_regimes == 0].index
    recession_dates_idx = recession_regimes[recession_regimes == 1].index
    
    non_recession_dates_df = result.loc[non_recession_dates_idx].copy()
    recession_dates_df= result.loc[recession_dates_idx].copy()
    
    print(f"\nFound {len(non_recession_dates_df)} non recession days")
    print(f"Found {len(recession_dates_df)} recession days")
    print(f"Total: {len(non_recession_dates_df) + len(recession_dates_df)} days")
    
    return recession_dates_df, non_recession_dates_df

def analyze_by_recession(recession_dates, non_recession_dates): 
    # Non recession Stats
    if len(non_recession_dates) > 0:
        nonrec_strat_ret = non_recession_dates['strategy_ret'].mean() * 252 * 100
        nonrec_strat_vol = non_recession_dates['strategy_ret'].std() * np.sqrt(252) * 100
        nonrec_strat_sharpe = non_recession_dates['strategy_ret'].mean() / non_recession_dates['strategy_ret'].std() * np.sqrt(252) if non_recession_dates['strategy_ret'].std() > 0 else 0
        nonrec_bh_ret = non_recession_dates['ret'].mean() * 252 * 100
        nonrec_bh_vol = non_recession_dates['ret'].std() * np.sqrt(252) * 100
        nonrec_bh_sharpe = non_recession_dates['ret'].mean() / non_recession_dates['ret'].std() * np.sqrt(252) if non_recession_dates['ret'].std() > 0 else 0
        nonrec_win = (non_recession_dates['strategy_ret'] > 0).sum() / len(non_recession_dates) * 100
        nonrec_strat_total = (1 + non_recession_dates['strategy_ret']).prod() - 1
        nonrec_bh_total = (1 + non_recession_dates['ret']).prod() - 1
    else:
        nonrec_strat_ret = nonrec_strat_vol = nonrec_strat_sharpe = 0
        nonrec_bh_ret = nonrec_bh_vol = nonrec_bh_sharpe = nonrec_win = 0
        nonrec_strat_total = nonrec_bh_total = 0
    
    # Recession Stats
    if len(recession_dates) > 0:
        rec_strat_ret = recession_dates['strategy_ret'].mean() * 252 * 100
        rec_strat_vol = recession_dates['strategy_ret'].std() * np.sqrt(252) * 100
        rec_strat_sharpe = recession_dates['strategy_ret'].mean() / recession_dates['strategy_ret'].std() * np.sqrt(252) if recession_dates['strategy_ret'].std() > 0 else 0
        rec_bh_ret = recession_dates['ret'].mean() * 252 * 100
        rec_bh_vol = recession_dates['ret'].std() * np.sqrt(252) * 100
        rec_bh_sharpe = recession_dates['ret'].mean() / recession_dates['ret'].std() * np.sqrt(252) if recession_dates['ret'].std() > 0 else 0
        rec_win = (recession_dates['strategy_ret'] > 0).sum() / len(recession_dates) * 100
        rec_strat_total = (1 + recession_dates['strategy_ret']).prod() - 1
        rec_bh_total = (1 + recession_dates['ret']).prod() - 1
    else:
        rec_strat_ret = rec_strat_vol = rec_strat_sharpe = 0
        rec_bh_ret = rec_bh_vol = rec_bh_sharpe = rec_win = 0
        rec_strat_total = rec_bh_total = 0
    
    return {
        'strategy total return in recession' : rec_strat_total,
        'strategy total return in nonrecession' : nonrec_strat_total,

        'buy n hold total return in recession' : rec_bh_total,
        'buy n hold total return in nonrecession' : nonrec_bh_total,

        'ann. strategy return in recession' : rec_strat_ret,
        'ann. strategy return in nonrecession' : nonrec_strat_ret,

        'ann. buy n hold return in recession' : rec_bh_ret,
        'ann. buy n hold return in nonrecession' : nonrec_bh_ret,

        'ann. strategy volatility in recession' : rec_strat_vol,
        'ann. strategy volatility in nonrecession' : nonrec_strat_vol,

        'ann. buy n hold volatility in recession' : rec_bh_vol,
        'ann. buy n hold volatility in nonrecession' : nonrec_bh_vol,

        'strategy sharpe ratio in recession' : rec_strat_sharpe,
        'strategy sharpe ratio in nonrecession' : nonrec_strat_sharpe,

        'buy n hold sharpe ratio in recession' : rec_strat_sharpe,
        'buy n hold sharpe ratio in nonrecession' : nonrec_strat_sharpe,

        'strategy winrate in recession' : rec_win,
        'strategy winrate in nonrecession' : nonrec_win
    }
    