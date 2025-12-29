import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

WINDOW = 5
DURATION = 10

def ATR(df, **params):
    atr_len = params['atr_len']
    
    high_low = df['High'] - df['Low']
    high_prev_close = (df['High'] - df['Close'].shift()).abs()
    low_prev_close = (df['Low'] - df['Close'].shift()).abs()

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_len).mean()
    return atr

def snr_strategy(df, **params):
    duration = params['duration']
    atr_buffer = params['atr_buffer']
    window = params['window']
    atr_params = params['atr'] 
    df = df.copy()

    df['ATR'] = ATR(df, **atr_params)

    # Save original index
    original_index = df.index.copy()
    
    # Reset to numeric index for processing
    df = df.reset_index(drop=True).copy()

    df['Support'] = 0
    df['Support_low_bound'] = np.nan
    df['Support_high_bound'] = np.nan
    df['Support_idx'] = np.nan

    df['Resistance'] = 0
    df['Resistance_low_bound'] = np.nan
    df['Resistance_high_bound'] = np.nan
    df['Resistance_idx'] = np.nan
    
    df['Bottom'] = 0
    rolling_min = df['Low'].rolling(window=window, center=False).min()  # Changed to center=False
    df['Bottom'] = np.where(df['Low'] == rolling_min, 1, 0)

    df['Top'] = 0
    rolling_max = df['High'].rolling(window=window, center=False).max()  # Changed to center=False
    df['Top'] = np.where(df['High'] == rolling_max, 1, 0)
    
    for i in range(len(df)):
        if i < duration:
            continue

        # Support
        past = df.iloc[i - duration:i]

        min_idx = past['Low'].idxmin()  
        min_candle = df.loc[min_idx]

        atr_val = df.loc[min_idx, 'ATR']

        low_bound = min_candle['Low'] - atr_buffer * atr_val
        high_bound = min_candle['Low'] + atr_buffer * atr_val

        df.iat[i, df.columns.get_loc('Support_low_bound')] = low_bound
        df.iat[i, df.columns.get_loc('Support_high_bound')] = high_bound
        df.iat[i, df.columns.get_loc('Support_idx')] = min_idx

        # Check if current price is touching the support zone
        cur_price = df.iloc[i]['Low']
        if low_bound <= cur_price <= high_bound:
            df.iat[i, df.columns.get_loc('Support')] = 1

        # Resistance
        max_idx = past['High'].idxmax()
        max_candle = df.loc[max_idx]

        atr_val = df.loc[max_idx, 'ATR']

        low_bound = max_candle['High'] - atr_buffer * atr_val
        high_bound = max_candle['High'] + atr_buffer * atr_val

        df.iat[i, df.columns.get_loc('Resistance_low_bound')] = low_bound
        df.iat[i, df.columns.get_loc('Resistance_high_bound')] = high_bound
        df.iat[i, df.columns.get_loc('Resistance_idx')] = max_idx

        # Check if current price is touching the resistance zone
        cur_price = df.iloc[i]['High']
        if low_bound <= cur_price <= high_bound:
            df.iat[i, df.columns.get_loc('Resistance')] = 1
    
    df['signal'] = 0
    df['signal'] = np.where(
        (df['Close'] >= df['Support_low_bound']) &
        (df['Close'] <= df['Support_high_bound']) &
        (df['Close'] > df['Close'].shift(1)),
        1,
        np.where(
            (df['Close'] >= df['Resistance_low_bound']) &
            (df['Close'] <= df['Resistance_high_bound']) &
            (df['Close'] < df['Close'].shift(1)),
            -1,
            0
        )
    )

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    # Create result with only needed columns
    result = df[['signal', 'strategy_ret', 'ret']].copy()
    
    # Restore the original date index
    result.index = original_index
    
    return result

def tsmom_strategy(df, **params):
    lookback = params['lookback']
    df = df.copy()
    df['past_ret'] = df['Close'] / df['Close'].shift(lookback) - 1
    df['signal'] = 0
    df['signal'] = np.where(df['past_ret'] > 0, 1,
                    np.where(df['past_ret'] < 0, -1, 0))


    df['signal'] = df['signal'].shift(1)
    df['signal'] = df['signal'].fillna(0)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    # Create result with only needed columns
    result = df[['signal', 'strategy_ret', 'ret']].copy()
    
    return result

def fvg_strategy(df):
    original_index = df.index.copy()
    df = df.reset_index(drop=True).copy()
    df['prev_2_high'] = df['High'].shift(2)
    df['Bull_FVG'] = (df['Low'] > df['prev_2_high']).astype(int)
    df['Bull_FVG_Val'] = (df['Low'] - df['prev_2_high']) * df['Bull_FVG'] / df['Close']
    df['Bull_FVG_High'] = df['prev_2_high'] * df['Bull_FVG']
    df['Bull_FVG_Low'] = df['Low'] * df['Bull_FVG']

    df['prev_2_low'] = df['Low'].shift(2)
    df['Bear_FVG'] = (df['High'] < df['prev_2_low']).astype(int)
    df['Bear_FVG_Val'] = (df['High'] - df['prev_2_low']) * df['Bear_FVG'] / df['Close']
    df['Bear_FVG_High'] = df['High'] * df['Bear_FVG']
    df['Bear_FVG_Low'] = df['prev_2_low'] * df['Bear_FVG']
    
    df['signal'] = 0
    df['FVG_Strategy'] = 0
    active_bull_high = np.nan
    active_bull_low = np.nan
    active_bear_high = np.nan
    active_bear_low = np.nan

    for i in range(len(df)):
        row = df.iloc[i]

        if row['Bull_FVG'] == 1:
            active_bull_high = row['Bull_FVG_High']
            active_bull_low = row['Bull_FVG_Low']

        if row['Bear_FVG'] == 1:
            active_bear_high = row['Bear_FVG_High']
            active_bear_low = row['Bear_FVG_Low']

        if not np.isnan(active_bull_high):
            if (row['Low'] <= active_bull_high) and (row['High'] >= active_bull_high):
                df.iat[i, df.columns.get_loc('FVG_Strategy')] = 1
                active_bull_high = active_bull_low = np.nan

        if not np.isnan(active_bear_low):
            if (row['High'] >= active_bear_low) and (row['Low'] <= active_bear_low):
                df.iat[i, df.columns.get_loc('FVG_Strategy')] = -1
                active_bear_high = active_bear_low = np.nan


    df['signal'] = df['FVG_Strategy']
    df['signal'] = df['signal'].shift(1)            # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    result = df[['signal','strategy_ret', 'ret']].copy()
    result.index = original_index

    return result

def bos_strategy(df, **params):
    duration = params['duration']
    atr_buffer = params['atr_buffer']
    window = params['window']
    lookback = params['lookback']
    atr_params = params['atr'] 
    df = df.copy()

    df['ATR'] = ATR(df, **atr_params)

    # Save original index
    original_index = df.index.copy()
    
    # Reset to numeric index for processing
    df = df.reset_index(drop=True).copy()
    
    df['Support'] = 0
    df['Support_low_bound'] = np.nan
    df['Support_high_bound'] = np.nan
    df['Support_idx'] = np.nan

    df['Resistance'] = 0
    df['Resistance_low_bound'] = np.nan
    df['Resistance_high_bound'] = np.nan
    df['Resistance_idx'] = np.nan

    df['Bottom'] = 0
    rolling_min = df['Low'].rolling(window=window, center=False).min()  # Changed to center=False
    df['Bottom'] = np.where(df['Low'] == rolling_min, 1, 0)

    df['Top'] = 0
    rolling_max = df['High'].rolling(window=window, center=False).max()  # Changed to center=False
    df['Top'] = np.where(df['High'] == rolling_max, 1, 0)
    
    for i in range(len(df)):
        if i < duration:
            continue

        # Support
        past = df.iloc[i - duration:i]

        min_idx = past['Low'].idxmin()  
        min_candle = df.loc[min_idx]

        atr_val = df.loc[min_idx, 'ATR']

        low_bound = min_candle['Low'] - atr_buffer * atr_val
        high_bound = min_candle['Low'] + atr_buffer * atr_val

        df.iat[i, df.columns.get_loc('Support_low_bound')] = low_bound
        df.iat[i, df.columns.get_loc('Support_high_bound')] = high_bound
        df.iat[i, df.columns.get_loc('Support_idx')] = min_idx

        # Check if current price is touching the support zone
        cur_price = df.iloc[i]['Low']
        if low_bound <= cur_price <= high_bound:
            df.iat[i, df.columns.get_loc('Support')] = 1

        # Resistance
        max_idx = past['High'].idxmax()
        max_candle = df.loc[max_idx]

        atr_val = df.loc[max_idx, 'ATR']

        low_bound = max_candle['High'] - atr_buffer * atr_val
        high_bound = max_candle['High'] + atr_buffer * atr_val

        df.iat[i, df.columns.get_loc('Resistance_low_bound')] = low_bound
        df.iat[i, df.columns.get_loc('Resistance_high_bound')] = high_bound
        df.iat[i, df.columns.get_loc('Resistance_idx')] = max_idx

        # Check if current price is touching the resistance zone
        cur_price = df.iloc[i]['High']
        if low_bound <= cur_price <= high_bound:
            df.iat[i, df.columns.get_loc('Resistance')] = 1

    df['BoS_Type'] = np.nan
    df['BoS_Confirmed'] = 0

    for i in range(lookback, len(df)):
        # Check if there was a support touch in recent history
        recent_support = df.iloc[i-lookback:i]['Support'].sum() > 0
        
        # Check if there was a resistance touch in recent history  
        recent_resistance = df.iloc[i-lookback:i]['Resistance'].sum() > 0
        
        if recent_support:
            # Look for break below support
            support_low = df.iloc[i-lookback:i][df['Support'] == 1]['Support_low_bound'].iloc[-1] if any(df.iloc[i-lookback:i]['Support'] == 1) else np.nan
            
            if not pd.isna(support_low) and df.iloc[i]['Close'] < support_low:
                df.iat[i, df.columns.get_loc('BoS_Type')] = 'Bearish'
                df.iat[i, df.columns.get_loc('BoS_Confirmed')] = 1
        
        if recent_resistance:
            # Look for break above resistance
            resistance_high = df.iloc[i-lookback:i][df['Resistance'] == 1]['Resistance_high_bound'].iloc[-1] if any(df.iloc[i-lookback:i]['Resistance'] == 1) else np.nan
            
            if not pd.isna(resistance_high) and df.iloc[i]['Close'] > resistance_high:
                df.iat[i, df.columns.get_loc('BoS_Type')] = 'Bullish'
                df.iat[i, df.columns.get_loc('BoS_Confirmed')] = 1
    
    df['signal'] = 0
    df['signal'] = np.where((df['BoS_Confirmed'] == 1) & (df['BoS_Type'] == 'Bullish'), 1, 
                        np.where((df['BoS_Confirmed'] == 1) & (df['BoS_Type'] == 'Bearish'), -1, 0))
    df['signal'] = df['signal'].shift(1)
    df['signal'] = df['signal'].fillna(0)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    # Create result with only needed columns
    result = df[['signal', 'strategy_ret', 'ret']].copy()
    
    # Restore the original date index
    result.index = original_index
    
    return result

def sma_long_strategy(df, **params):
    fast = params['fast']
    slow = params['slow']
    df = df.copy()
    df['fast'] = df['Close'].rolling(fast).mean()
    df['slow'] = df['Close'].rolling(slow).mean()
    
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1   # long
    df['signal'] = df['signal'].shift(1)            # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    result = df[['signal', 'strategy_ret', 'ret']].copy()

    return result

def sma_short_strategy(df, **params):
    fast = params['fast']
    slow = params['slow']
    df = df.copy()
    df['fast'] = df['Close'].rolling(fast).mean()
    df['slow'] = df['Close'].rolling(slow).mean()
    
    df['signal'] = 0
    df.loc[df['fast'] < df['slow'], 'signal'] = -1   # short
    df['signal'] = df['signal'].shift(1)            # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)
    df['strategy_ret'] = df['signal'] * df['ret']

    # strategy returns
    result = df[['signal', 'strategy_ret', 'ret']].copy()

    return result

def sma_combined_strategy(df, **params):
    fast = params['fast']
    slow = params['slow']
    df = df.copy()
    df['fast'] = df['Close'].rolling(fast).mean()
    df['slow'] = df['Close'].rolling(slow).mean()
    
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1   # long
    df.loc[df['fast'] < df['slow'], 'signal'] = -1  # short
    df['signal'] = df['signal'].shift(1)            # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    result = df[['signal', 'strategy_ret', 'ret']].copy()

    return result

def macd_strategy(df, **params):
    fast = params['fast']
    slow = params['slow']
    signal = params['signal']
    df = df.copy()
    df[f'{fast}_EMA'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df[f'{slow}_EMA'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df[f'{fast}_EMA'] - df[f'{slow}_EMA']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['signal'] = 0
    df['MACD_Strategy'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)   # short
    df['signal'] = df['MACD_Strategy']
    df['signal'] = df['signal'].shift(1)                         # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    result = df[['signal', 'strategy_ret', 'ret']].copy()

    return result

def bbnrsi_strategy(df, **params):
    overbought = params['rsi_overbought']
    oversold = params['rsi_oversold']
    period = params['rsi_period']
    window = params['bb_window']
    num_std = params['bb_std']
    df = df.copy()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's smoothing
    avg_gain = avg_gain.shift(1).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.shift(1).ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi.fillna(50)

    df['BB_MA'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (num_std * df['BB_Std'])
    df['BB_Lower'] = df['BB_MA'] - (num_std * df['BB_Std'])

    df['RSI_Strategy'] = 0
    df.loc[df['RSI'] > overbought, 'RSI_Strategy'] = -1
    df.loc[df['RSI'] < oversold, 'RSI_Strategy'] = 1
    
    df['BB_Strategy'] = 0
    df.loc[df['Close'] > df['BB_Upper'], 'BB_Strategy'] = -1
    df.loc[df['Close'] < df['BB_Lower'], 'BB_Strategy'] = 1

    df['signal'] = 0

    agree_long  = (df['RSI_Strategy'] == 1) & (df['BB_Strategy'] == 1)
    agree_short = (df['RSI_Strategy'] == -1) & (df['BB_Strategy'] == -1)
    
    # 1. If both indicators agree
    df.loc[agree_long, 'signal'] = 1
    df.loc[agree_short, 'signal'] = -1

    # 2. If they disagree but RSI has a signal
    df.loc[(df['signal'] == 0) & (df['RSI_Strategy'] != 0), 'signal'] = df['RSI_Strategy']

    df['signal'] = df['signal'].shift(1)            # trade next day
    df['signal'].fillna(0, inplace=True)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # strategy returns
    df['strategy_ret'] = df['signal'] * df['ret']

    result = df[['signal','strategy_ret', 'ret']].copy()

    return result
