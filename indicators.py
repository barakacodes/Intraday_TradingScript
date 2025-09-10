"""
Vectorized technical indicators module.

Provides functions that accept pandas Series or DataFrames and return pandas Series
(or tuples of Series) computed in a vectorized way (no per-row apply).

Functions:
- rsi(close, period=14)
- atr(df, period=14)            # df must have columns: high, low, close
- bollinger_bands(close, period=20, num_std=2) -> (upper, middle, lower)
- stochastic(df, k_period=14, d_period=3) -> (k_series, d_series)
- adx(df, period=14)            # df must have columns: high, low, close
- macd(close, fast=12, slow=26, signal=9) -> (macd_line, signal_line, hist)
- add_basic_features(df)        # convenience to add common features to df in-place
"""

from typing import Tuple
import pandas as pd
import numpy as np

# Try to use pandas_ta if available (faster / battle-tested). If not available, use pure-pandas.
try:
    import pandas_ta as pta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized RSI (Wilder's smoothing). Returns a Series aligned with close."""
    if _HAS_PANDAS_TA:
        return pta.rsi(close, length=period)
    # Pure pandas implementation (Wilder)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing: exponential-like with alpha=1/period (using .ewm with adjust=False)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range and ATR (simple Wilder smoothing). df must contain 'high','low','close'"""
    if _HAS_PANDAS_TA:
        return pta.atr(df['high'], df['low'], df['close'], length=period)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing via ewm with alpha=1/period and adjust=False approximates Wilder's smoothing
    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr_series.fillna(method='backfill').fillna(0.0)


def bollinger_bands(close: pd.Series, period: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_ma, lower_band) as Series."""
    if _HAS_PANDAS_TA:
        bb = pta.bbands(close, length=period, std=num_std)
        try:
            upper = bb[f'BBU_{period}_{float(num_std)}']
            middle = bb[f'BBM_{period}_{float(num_std)}']
            lower = bb[f'BBL_{period}_{float(num_std)}']
            return upper, middle, lower
        except Exception:
            cols = bb.columns.tolist()
            return bb[cols[-3]], bb[cols[-2]], bb[cols[-1]]
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper.fillna(close), ma.fillna(close), lower.fillna(close)


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Fast %K and %D Stochastic oscillator as Series. df requires 'high','low','close'."""
    if _HAS_PANDAS_TA:
        stoch = pta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period)
        for kname in ['STOCHk_14_3_3', f'STOCHk_{k_period}_{d_period}_3', f'STOCHk_{k_period}_{d_period}_3']:
            if kname in stoch.columns:
                k_series = stoch[kname]
                break
        else:
            k_series = stoch.iloc[:, 0]
        d_series = stoch.iloc[:, -1]
        return k_series.fillna(50.0), d_series.fillna(50.0)
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    k = k.fillna(50.0)
    d = d.fillna(50.0)
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) as Series. df requires 'high','low','close'."""
    if _HAS_PANDAS_TA:
        return pta.adx(df['high'], df['low'], df['close'], length=period)['ADX_14']
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period).sum()
    plus_dm_sum = plus_dm.rolling(window=period).sum()
    minus_dm_sum = minus_dm.rolling(window=period).sum()

    plus_di = 100 * (plus_dm_sum / atr_series).replace([np.inf, -np.inf], 0).fillna(0)
    minus_di = 100 * (minus_dm_sum / atr_series).replace([np.inf, -np.inf], 0).fillna(0)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.rolling(window=period).mean().fillna(20.0)
    return adx_series.fillna(20.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return macd_line, signal_line, histogram"""
    if _HAS_PANDAS_TA:
        macd_df = pta.macd(close, fast=fast, slow=slow, signal=signal)
        try:
            macd_line = macd_df[f'MACD_{fast}_{slow}_{signal}']
            hist = macd_df[f'MACDh_{fast}_{slow}_{signal}']
            signal_line = macd_df[f'MACDs_{fast}_{slow}_{signal}']
            return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)
        except Exception:
            cols = macd_df.columns
            if len(cols) >= 3:
                return macd_df.iloc[:,0].fillna(0.0), macd_df.iloc[:,2].fillna(0.0), macd_df.iloc[:,1].fillna(0.0)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)


def add_basic_features(df: pd.DataFrame, rsi_period=14, atr_period=14, bb_period=20, stoch_k=14, stoch_d=3):
    """Add common indicator columns to df in-place. Assumes df has 'high','low','close','time'."""
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df['rsi'] = rsi(df['close'], period=rsi_period)
    df['atr'] = atr(df, period=atr_period)
    bb_u, bb_m, bb_l = bollinger_bands(df['close'], period=bb_period)
    df['bb_upper'] = bb_u
    df['bb_middle'] = bb_m
    df['bb_lower'] = bb_l
    k, d = stochastic(df, k_period=stoch_k, d_period=stoch_d)
    df['stoch_k'] = k
    df['stoch_d'] = d
    df['adx'] = adx(df)
    macd_line, signal_line, hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['lag1'] = df['close'].shift(1)
    df['lag2'] = df['close'].shift(2)
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
    if 'open' in df.columns:
        df['body'] = (df['close'] - df['open']).abs()
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
        df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']
    return df