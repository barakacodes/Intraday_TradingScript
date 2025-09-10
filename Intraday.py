import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import json

# -------------------------
# CONFIGURATION
# -------------------------
logging.basicConfig(
    filename='hfm_trading_bot.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

DEFAULT_PARAMS = {
    'LOOKBACK': 60,
    'RSI_PERIOD': 14,
    'ATR_PERIOD': 14,
    'VOLATILITY_PERIOD': 20,
    'MIN_ACCOUNT_BALANCE': 5.0,
    'ATR_MULTIPLIER_TP': 15.0,
    'BARS_TO_FETCH': 3000,  # Increased
    'LOOP_SLEEP': 600,
    'MIN_LOT': 0.01,
    'MAX_LOT': 1.0,
    'MAX_DRAWDOWN_PERCENT': 2.0,
    'MAX_OPEN_TRADES': 12,
    'MAX_TRADES_PER_SYMBOL': 2,
    'MAX_TRADE_DURATION_HOURS': 24,
    'ANALYSIS_BARS': 30,  # Increased for Stochastic
    'CONFIDENCE_THRESHOLD': 0.5,  # Lowered to allow MA trades
    'DEFAULT_STOPS_LEVEL': 10,
    'DEFAULT_THRESHOLD': 0.0001,
}

SYMBOL_RISK = {
    "XAUUSDm": 0.005,
    "BTCUSDm": 0.002,
    "USDJPYm": 0.01,
    "EURUSDm": 0.01,
    "US30m": 0.005,
}

SYMBOL_TP_MULTIPLIERS = {
    "XAUUSDm": 25.0,
    "BTCUSDm": 30.0,
    "USDJPYm": 20.0,
    "EURUSDm": 20.0,
    "US30m": 25.0,
}

MIN_TP_DISTANCE = {
    "XAUUSDm": 15.0,
    "BTCUSDm": 1000.0,  # Increased
    "USDJPYm": 0.10,
    "EURUSDm": 0.0010,
    "US30m": 10.0,
}

BREAK_EVEN_THRESHOLDS = {
    "XAUUSDm": 25.0,
    "BTCUSDm": 50.0,
    "USDJPYm": 15.0,
    "EURUSDm": 15.0,
    "US30m": 20.0,
}

BREAK_EVEN_SECURE_PERCENT = 0.5
SYMBOLS = ["XAUUSDm", "BTCUSDm", "USDJPYm", "EURUSDm", "US30m"]
ACCOUNT = 211071607
PASSWORD = "Br1ll1ant$"
SERVER = "Exness-MT5Trial9"
RF_MODELS = {}
ENABLE_EMAIL_ALERTS = False
PREV_ANALYSIS = {}

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def safe_format(value, fmt):
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (ValueError, TypeError):
        return str(value)

def send_alert(subject, body):
    if not ENABLE_EMAIL_ALERTS:
        logging.debug(f"Email alerts disabled, skipping: {subject}")
        return
    try:
        import smtplib
        from email.message import EmailMessage
        EMAIL_CONFIG = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 465,
            'email': 'your_email@gmail.com',
            'password': 'your_app_password'
        }
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = EMAIL_CONFIG['email']
        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as smtp:
            smtp.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            smtp.send_message(msg)
        logging.info(f"Alert sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")

def ensure_mt5_connection():
    try:
        if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
            logging.error(f"MT5 connection failed: {mt5.last_error()}")
            send_alert("MT5 Connection Failed", f"MT5 connection failed: {mt5.last_error()}")
            return False
        logging.debug("MT5 connection verified")
        return True
    except Exception as e:
        logging.error(f"Error in ensure_mt5_connection: {e}")
        send_alert("MT5 Connection Error", f"Error in MT5 connection: {e}")
        return False

def log_account_status():
    try:
        account_info = mt5.account_info()
        if not account_info:
            logging.error("Failed to get account info")
            return
        total_positions = mt5.positions_total()
        logging.info(f"Account Status: balance={safe_format(account_info.balance, '.2f')}, "
                     f"equity={safe_format(account_info.equity, '.2f')}, "
                     f"margin_free={safe_format(account_info.margin_free, '.2f')}, "
                     f"open_positions={total_positions}")
        open_trades = mt5.positions_get()
        if open_trades:
            for pos in open_trades:
                if pos.magic == 234000:
                    logging.info(f"Open Trade: symbol={pos.symbol}, ticket={pos.ticket}, "
                                 f"type={'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}, "
                                 f"entry_price={safe_format(pos.price_open, '.5f')}, "
                                 f"profit={safe_format(pos.profit, '.2f')}")
    except Exception as e:
        logging.error(f"Error in log_account_status: {e}")

# -------------------------
# TECHNICAL INDICATORS
# -------------------------
def calculate_rsi(df, period=14):
    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        logging.debug(f"RSI calculated for {symbol}: {rsi:.2f}")
        return rsi if not np.isnan(rsi) else 50.0
    except Exception as e:
        logging.error(f"Error in calculate_rsi: {e}")
        return 50.0

def calculate_atr(df, period=14, symbol_point=0.0001):
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        logging.debug(f"ATR calculated for {symbol}: {atr:.5f}")
        return atr if not np.isnan(atr) else symbol_point
    except Exception as e:
        logging.error(f"Error in calculate_atr: {e}")
        return symbol_point

def calculate_bollinger_bands(df, period=20, num_std=2):
    try:
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        logging.debug(f"Bollinger Bands for {symbol}: upper={upper.iloc[-1]:.5f}, lower={lower.iloc[-1]:.5f}")
        u = upper.iloc[-1]
        l = lower.iloc[-1]
        return (u if not np.isnan(u) else df['close'].iloc[-1], l if not np.isnan(l) else df['close'].iloc[-1])
    except Exception as e:
        logging.error(f"Error in calculate_bollinger_bands: {e}")
        close = df['close'].iloc[-1] if len(df) > 0 else 0.0
        return close, close

def calculate_stochastic(df, k_period=14, d_period=3):
    try:
        if len(df) < k_period:
            logging.warning(f"Insufficient data for Stochastic calculation: {len(df)} rows, need {k_period}")
            return None, None
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        k_val = k.iloc[-1]
        d_val = d.iloc[-1]
        if pd.isna(k_val) or pd.isna(d_val):
            logging.warning(f"Stochastic calculation returned NaN for {symbol}")
            return None, None
        logging.debug(f"Stochastic for {symbol}: %K={k_val:.2f}, %D={d_val:.2f}")
        return k_val, d_val
    except Exception as e:
        logging.error(f"Error in calculate_stochastic: {e}")
        return None, None

def calculate_adx(df, period=14):
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = tr.rolling(window=period).sum()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr)
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean().iloc[-1]
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        logging.debug(f"ADX calculated for {symbol}: {adx:.2f}")
        return adx if not np.isnan(adx) else 20.0
    except Exception as e:
        logging.error(f"Error in calculate_adx: {e}")
        return 20.0

# -------------------------
# MARKET ANALYSIS
# -------------------------
def analyze_market_conditions(symbol, df):
    try:
        logging.info(f"Starting market analysis for {symbol}")
        rsi = calculate_rsi(df, DEFAULT_PARAMS['RSI_PERIOD'])
        atr = calculate_atr(df, DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point)
        vol = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().iloc[-1]
        vol = 0.0 if pd.isna(vol) else vol
        macd = df['close'].ewm(span=12).mean().iloc[-1] - df['close'].ewm(span=26).mean().iloc[-1]
        macd = 0.0 if pd.isna(macd) else macd
        bb_upper, bb_lower = calculate_bollinger_bands(df)
        stoch_k, stoch_d = calculate_stochastic(df)
        adx = calculate_adx(df)
        current_price = df['close'].iloc[-1]

        retrain = False
        prev = PREV_ANALYSIS.get(symbol, {})
        if prev:
            prev_adx = prev.get('adx', 0)
            prev_vol = prev.get('vol', 0)
            if (adx is not None and prev_adx and abs(adx - prev_adx) / prev_adx > 0.2) or \
               (vol is not None and prev_vol and abs(vol - prev_vol) / prev_vol > 0.2):
                retrain = True
        else:
            retrain = True

        if retrain and len(df) >= DEFAULT_PARAMS['ANALYSIS_BARS']:
            RF_MODELS[symbol] = train_rf_model(symbol, df)
            if RF_MODELS[symbol] is None:
                logging.warning(f"Model retraining failed for {symbol}, using existing model")
            else:
                logging.info(f"Model retrained for {symbol} due to significant market change")

        if symbol not in RF_MODELS or RF_MODELS[symbol] is None:
            logging.warning(f"No RF model for {symbol} during analysis, using default threshold")
            return None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None

        features = pd.DataFrame({
            'rsi': [rsi],
            'atr': [atr],
            'vol': [vol],
            'macd': [macd],
            'bb_upper': [bb_upper],
            'bb_lower': [bb_lower],
            'lag1': [current_price],
            'lag2': [df['close'].iloc[-2] if len(df) > 1 else current_price],
            'stoch_k': [stoch_k if stoch_k is not None else 50.0],
            'stoch_d': [stoch_d if stoch_d is not None else 50.0],
            'adx': [adx],
            'hour': [df['time'].iloc[-1].hour],
            'day_of_week': [df['time'].iloc[-1].dayofweek]
        })

        model = RF_MODELS[symbol]
        pred_class = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        confidence = max(pred_proba)

        feature_names = features.columns
        importance = model.feature_importances_
        importance_log = {name: imp for name, imp in zip(feature_names, importance)}
        logging.debug(f"Feature importance for {symbol}: {importance_log}")

        if pred_class == 1:
            pred_price = current_price * 1.01
            signal = "BUY"
        elif pred_class == -1:
            pred_price = current_price * 0.99
            signal = "SELL"
        else:
            pred_price = current_price
            signal = "NO TRADE"

        base_threshold = max(0.0001, atr / current_price * 0.5 if atr and current_price else 0.0001)
        if adx is not None and adx < 20:
            threshold_adjustment = 1.5
        elif adx is not None and adx > 25:
            threshold_adjustment = 0.8
        else:
            threshold_adjustment = 1.0
        if vol is not None and vol > df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().mean() * 1.5:
            threshold_adjustment *= 1.2
        dynamic_threshold = base_threshold * threshold_adjustment

        stoch_signal = None
        if stoch_k is not None and stoch_d is not None and len(df) > 1:
            prev_k, prev_d = calculate_stochastic(df.iloc[:-1], DEFAULT_PARAMS['RSI_PERIOD'], 3)
            if prev_k is not None and prev_d is not None:
                if prev_k < prev_d and stoch_k > stoch_d and stoch_k < 80:
                    stoch_signal = "BUY"
                elif prev_k > prev_d and stoch_k < stoch_d and stoch_k > 20:
                    stoch_signal = "SELL"
            else:
                logging.warning(f"Stochastic signal calculation failed for {symbol}, prev_k={prev_k}, prev_d={prev_d}")
        else:
            logging.warning(f"Stochastic signal not computed for {symbol}, stoch_k={stoch_k}, stoch_d={stoch_d}")

        PREV_ANALYSIS[symbol] = {'adx': adx, 'vol': vol}
        logging.info(f"Market Analysis for {symbol}: predicted_price={safe_format(pred_price, '.5f')}, signal={signal}, "
                     f"confidence={safe_format(confidence, '.2f')}, adx={safe_format(adx, '.2f')}, volatility={safe_format(vol, '.6f')}, "
                     f"rsi={safe_format(rsi, '.2f')}, stoch_k={safe_format(stoch_k, '.2f')}, stoch_d={safe_format(stoch_d, '.2f')}, "
                     f"stoch_signal={stoch_signal}, dynamic_threshold={safe_format(dynamic_threshold, '.4%')}")
        return pred_price, signal, dynamic_threshold, confidence, stoch_signal, adx, stoch_k, stoch_d
    except Exception as e:
        logging.error(f"Error in analyze_market_conditions for {symbol}: {e}")
        return None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None

# -------------------------
# ML MODEL TRAINING
# -------------------------
def train_rf_model(symbol, df, retries=5):
    for attempt in range(retries):
        try:
            logging.info(f"Starting RF model training for {symbol}, attempt {attempt+1}")
            df = df.copy()
            logging.debug(f"Training data rows for {symbol}: {len(df)}")
            if len(df) < DEFAULT_PARAMS['RSI_PERIOD']:
                logging.warning(f"Insufficient data rows for feature calculation: {len(df)} < {DEFAULT_PARAMS['RSI_PERIOD']}")
                time.sleep(10)
                continue

            features = ['rsi', 'atr', 'vol', 'macd', 'bb_upper', 'bb_lower', 'lag1', 'lag2', 'stoch_k', 'stoch_d', 'adx', 'hour', 'day_of_week']
            df_dict = {'symbol': symbol, 'high': df['high'], 'low': df['low'], 'close': df['close'], 'time': df['time']}

            try:
                df['rsi'] = df['close'].diff().rolling(DEFAULT_PARAMS['RSI_PERIOD']).apply(
                    lambda x: calculate_rsi(pd.DataFrame({'close': x, 'symbol': symbol}), DEFAULT_PARAMS['RSI_PERIOD'])
                )
            except Exception as e:
                logging.warning(f"RSI calculation failed for {symbol}: {e}, using default")
                df['rsi'] = 50.0

            try:
                df['atr'] = df.apply(
                    lambda x: calculate_atr(pd.DataFrame(df_dict), DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.0001),
                    axis=1
                )
            except Exception as e:
                logging.warning(f"ATR calculation failed for {symbol}: {e}, using default")
                df['atr'] = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.0001

            try:
                df['vol'] = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std()
            except Exception as e:
                logging.warning(f"Volatility calculation failed for {symbol}: {e}, using default")
                df['vol'] = 0.0

            try:
                df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            except Exception as e:
                logging.warning(f"MACD calculation failed for {symbol}: {e}, using default")
                df['macd'] = 0.0

            try:
                df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
            except Exception as e:
                logging.warning(f"Bollinger Bands calculation failed for {symbol}: {e}, using default")
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']

            df['lag1'] = df['close'].shift(1)
            df['lag2'] = df['close'].shift(2)

            try:
                df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
            except Exception as e:
                logging.warning(f"Stochastic calculation failed for {symbol}: {e}, using default")
                df['stoch_k'] = 50.0
                df['stoch_d'] = 50.0

            try:
                df['adx'] = calculate_adx(df)
            except Exception as e:
                logging.warning(f"ADX calculation failed for {symbol}: {e}, using default")
                df['adx'] = 20.0

            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek

            df['future_return'] = df['close'].shift(-1) / df['close'] - 1
            df['target'] = 0
            df.loc[df['future_return'] >= 0.01, 'target'] = 1
            df.loc[df['future_return'] <= -0.01, 'target'] = -1

            df_buy = df[df['target'] == 1]
            df_sell = df[df['target'] == -1]
            df_no_trade = df[df['target'] == 0].sample(n=min(len(df_buy), len(df_sell)) * 2, random_state=42, replace=True)
            df = pd.concat([df_buy, df_sell, df_no_trade])
            logging.info(f"Training data for {symbol}: {len(df_buy)} buy, {len(df_sell)} sell, {len(df_no_trade)} no trade")

            df = df.dropna(subset=features + ['target'])
            if len(df) < 30:
                logging.warning(f"Insufficient data to train RF model for {symbol}: {len(df)} rows")
                time.sleep(10)
                continue

            X = df[features]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            param_grid = {'n_estimators': [50], 'max_depth': [5]}
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"RF model trained for {symbol}, test accuracy: {accuracy:.4f}, best params: {grid_search.best_params_}")
            return best_model
        except Exception as e:
            logging.error(f"Error training RF model for {symbol} on attempt {attempt+1}: {e}")
            time.sleep(10)
    logging.error(f"Failed to train RF model for {symbol} after {retries} attempts")
    return None

# -------------------------
# PREDICTION LOGIC
# -------------------------
def predict_next_price(df, symbol, dynamic_threshold):
    try:
        logging.debug(f"Starting prediction for {symbol}")
        rsi = calculate_rsi(df, DEFAULT_PARAMS['RSI_PERIOD'])
        atr = calculate_atr(df, DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point)
        vol = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().iloc[-1]
        vol = 0.0 if pd.isna(vol) else vol
        macd = df['close'].ewm(span=12).mean().iloc[-1] - df['close'].ewm(span=26).mean().iloc[-1]
        macd = 0.0 if pd.isna(macd) else macd
        bb_upper, bb_lower = calculate_bollinger_bands(df)
        lag1 = df['close'].iloc[-1]
        lag2 = df['close'].iloc[-2] if len(df) > 1 else lag1
        stoch_k, stoch_d = calculate_stochastic(df)
        adx = calculate_adx(df)
        hour = df['time'].iloc[-1].hour
        day_of_week = df['time'].iloc[-1].dayofweek
        current_price = df['close'].iloc[-1]

        if dynamic_threshold is None or np.isnan(dynamic_threshold):
            dynamic_threshold = max(0.0001, atr / current_price * 0.5 if atr and current_price else DEFAULT_PARAMS['DEFAULT_THRESHOLD'])
            logging.debug(f"Set default dynamic_threshold for {symbol}: {dynamic_threshold}")

        if symbol not in RF_MODELS or RF_MODELS[symbol] is None:
            logging.warning(f"No RF model for {symbol}, falling back to MA crossover")
            short_ma = df['close'].rolling(20).mean().iloc[-1]
            long_ma = df['close'].rolling(100).mean().iloc[-1]
            if short_ma > long_ma:
                pred_price = current_price * 1.01
                confidence = 0.5
                signal = "BUY"
                logging.info(f"{symbol} predicted BUY (MA): pred_price={safe_format(pred_price, '.5f')}, confidence={safe_format(confidence, '.2f')}")
            else:
                pred_price = current_price * 0.99
                confidence = 0.5
                signal = "SELL"
                logging.info(f"{symbol} predicted SELL (MA): pred_price={safe_format(pred_price, '.5f')}, confidence={safe_format(confidence, '.2f')}")
            base_threshold = max(0.0001, atr / current_price * 0.5 if atr and current_price else DEFAULT_PARAMS['DEFAULT_THRESHOLD'])
            if adx is not None and adx < 20:
                threshold_adjustment = 1.5
            elif adx is not None and adx > 25:
                threshold_adjustment = 0.8
            else:
                threshold_adjustment = 1.0
            if vol is not None and vol > df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().mean() * 1.5:
                threshold_adjustment *= 1.2
            dynamic_threshold = base_threshold * threshold_adjustment
            return pred_price, rsi, vol, macd, bb_upper, bb_lower, atr, dynamic_threshold, confidence, signal, adx, stoch_k, stoch_d

        features = pd.DataFrame({
            'rsi': [rsi],
            'atr': [atr],
            'vol': [vol],
            'macd': [macd],
            'bb_upper': [bb_upper],
            'bb_lower': [bb_lower],
            'lag1': [lag1],
            'lag2': [lag2],
            'stoch_k': [stoch_k if stoch_k is not None else 50.0],
            'stoch_d': [stoch_d if stoch_d is not None else 50.0],
            'adx': [adx],
            'hour': [hour],
            'day_of_week': [day_of_week]
        })

        model = RF_MODELS[symbol]
        pred_class = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        confidence = max(pred_proba)

        if pred_class == 1:
            pred_price = current_price * 1.01
            signal = "BUY"
        elif pred_class == -1:
            pred_price = current_price * 0.99
            signal = "SELL"
        else:
            pred_price = current_price
            signal = "NO TRADE"

        feature_names = features.columns
        importance = model.feature_importances_
        importance_log = {name: imp for name, imp in zip(feature_names, importance)}
        logging.debug(f"Feature importance for {symbol} prediction: {importance_log}")

        logging.info(f"{symbol} predicted {signal} (RF): pred_price={safe_format(pred_price, '.5f')}, class={pred_class}, confidence={safe_format(confidence, '.2f')}")
        return pred_price, rsi, vol, macd, bb_upper, bb_lower, atr, dynamic_threshold, confidence, signal, adx, stoch_k, stoch_d
    except Exception as e:
        logging.error(f"Error in predict_next_price for {symbol}: {e}")
        return None, None, None, None, None, None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None

# -------------------------
# DATA FETCHING
# -------------------------
def get_historical_data(symbol, timeframe, bars=1000, retries=5):
    for attempt in range(retries):
        try:
            logging.debug(f"Fetching historical data for {symbol}, timeframe={timeframe}, bars={bars}, attempt={attempt+1}")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                logging.warning(f"No historical data for {symbol}: {mt5.last_error()}")
                time.sleep(10)
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            logging.debug(f"Historical data fetched for {symbol}: {len(df)} bars")
            return df
        except Exception as e:
            logging.error(f"Error in get_historical_data for {symbol}: {e}")
            time.sleep(10)
    logging.error(f"Failed to fetch historical data for {symbol} after {retries} attempts")
    return None

# -------------------------
# ORDER MANAGEMENT
# -------------------------
def place_order_no_sl(symbol, order_type, volume, price, tp_price=None):
    try:
        logging.debug(f"Placing order for {symbol}: type={order_type}, volume={volume}, price={price}, tp={tp_price}")
        info = mt5.symbol_info(symbol)
        if not info:
            logging.error(f"No symbol info for {symbol}")
            return None
        if volume < info.volume_min or volume > info.volume_max:
            logging.error(f"Invalid lot size {volume} for {symbol}: min={info.volume_min}, max={info.volume_max}")
            return None
        filling_mode = info.filling_mode
        logging.debug(f"{symbol} filling modes: {filling_mode}")
        if filling_mode & mt5.ORDER_FILLING_IOC:
            type_filling = mt5.ORDER_FILLING_IOC
        elif filling_mode & mt5.ORDER_FILLING_FOK:
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & mt5.ORDER_FILLING_RETURN:
            type_filling = mt5.ORDER_FILLING_RETURN
        else:
            logging.error(f"No supported filling mode for {symbol}")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.time < int(time.time()) - 60:
            logging.error(f"No recent tick data for {symbol}, market may be closed")
            return None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "deviation": 20,
            "sl": 0.0,
            "tp": float(tp_price) if tp_price is not None else 0.0,
            "magic": 234000,
            "comment": "NN-HFM-no-sl",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }
        logging.debug(f"Order request: {request}")
        res = mt5.order_send(request)
        if res is None:
            logging.error(f"order_send returned None for {symbol}, mt5.last_error={mt5.last_error()}")
            return None
        logging.debug(f"order_send response for {symbol}: {res}")
        if hasattr(res, 'retcode') and res.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = getattr(res, 'order', None)
            logging.info(f"Order placed for {symbol}: ticket={ticket}, type={order_type}, volume={volume}, price={price}, tp={tp_price}")
            return int(ticket) if ticket else None
        else:
            logging.error(f"Order failed for {symbol}: retcode={res.retcode}, comment={getattr(res, 'comment', 'N/A')}")
            return None
    except Exception as e:
        logging.error(f"Error in place_order_no_sl for {symbol}: {e}")
        return None

# -------------------------
# TRADE MANAGEMENT
# -------------------------
def manage_breakeven(symbol):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            logging.debug(f"No open positions for {symbol}")
            return

        info = mt5.symbol_info(symbol)
        if not info:
            logging.error(f"No symbol info for {symbol} in manage_breakeven")
            return

        profit_threshold = BREAK_EVEN_THRESHOLDS.get(symbol, 10.0)
        secure_percent = BREAK_EVEN_SECURE_PERCENT
        stops_level = getattr(info, 'stops_level', DEFAULT_PARAMS['DEFAULT_STOPS_LEVEL']) * info.point
        if not hasattr(info, 'stops_level'):
            logging.warning(f"No stops_level for {symbol}, using default: {DEFAULT_PARAMS['DEFAULT_STOPS_LEVEL']} points")

        for pos in positions:
            if pos.magic != 234000:
                continue
            logging.info(f"Open position for {symbol}: ticket={pos.ticket}, type={'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}, "
                         f"entry_price={safe_format(pos.price_open, '.5f')}, current_price={safe_format(pos.price_current, '.5f')}, "
                         f"profit={safe_format(pos.profit, '.2f')}, sl={safe_format(pos.sl, '.5f')}, tp={safe_format(pos.tp, '.5f')}")

            if pos.sl == 0.0 and pos.profit >= profit_threshold:
                secure_profit = pos.profit * secure_percent
                if pos.type == mt5.ORDER_TYPE_BUY:
                    contract_size = info.trade_contract_size or 100000
                    price_adjustment = secure_profit / (pos.volume * contract_size)
                    breakeven_sl = pos.price_open + price_adjustment
                    if pos.price_current - breakeven_sl < stops_level:
                        breakeven_sl = pos.price_current - stops_level
                else:
                    contract_size = info.trade_contract_size or 100000
                    price_adjustment = secure_profit / (pos.volume * contract_size)
                    breakeven_sl = pos.price_open - price_adjustment
                    if breakeven_sl - pos.price_current < stops_level:
                        breakeven_sl = pos.price_current + stops_level

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": symbol,
                    "sl": float(breakeven_sl),
                    "tp": float(pos.tp),
                }
                res = mt5.order_send(request)
                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to set breakeven for {symbol} ticket={pos.ticket}: {mt5.last_error()}")
                else:
                    logging.info(f"Breakeven set for {symbol} ticket={pos.ticket}: sl={safe_format(breakeven_sl, '.5f')}, secured_profit={safe_format(secure_profit, '.2f')}")
    except Exception as e:
        logging.error(f"Error in manage_breakeven for {symbol}: {e}")

def close_old_trades(symbol):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        max_duration = timedelta(hours=DEFAULT_PARAMS['MAX_TRADE_DURATION_HOURS'])
        current_time = datetime.now()

        for pos in positions:
            if pos.magic != 234000:
                continue
            open_time = datetime.fromtimestamp(pos.time)
            if current_time - open_time > max_duration and pos.profit <= 0:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Close old trade",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = mt5.order_send(request)
                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to close old trade for {symbol} ticket={pos.ticket}: {mt5.last_error()}")
                else:
                    logging.info(f"Closed old trade for {symbol}: ticket={pos.ticket}, profit={safe_format(pos.profit, '.2f')}")
    except Exception as e:
        logging.error(f"Error in close_old_trades for {symbol}: {e}")

# -------------------------
# LOT SIZE CALCULATION
# -------------------------
def calculate_dynamic_lot_size(symbol, balance, atr, point):
    try:
        info = mt5.symbol_info(symbol)
        if not info:
            logging.error(f"No symbol info for {symbol} in lot size calc")
            return DEFAULT_PARAMS['MIN_LOT']
        risk_percent = SYMBOL_RISK.get(symbol, 0.01)
        risk_amount = balance * risk_percent
        contract_size = info.trade_contract_size or 100000
        pip_value = contract_size * point
        stop_distance = atr if atr > 0 else point * 10
        stop_value = (stop_distance / point) * pip_value
        if stop_value <= 0:
            logging.warning(f"Invalid stop value for {symbol}, using min lot")
            return DEFAULT_PARAMS['MIN_LOT']
        lots = risk_amount / stop_value
        lots = max(info.volume_min, min(lots, info.volume_max))
        step = info.volume_step or 0.01
        lots = round(lots / step) * step
        logging.debug(f"Lot size for {symbol}: {lots}")
        return lots
    except Exception as e:
        logging.error(f"Error in calculate_dynamic_lot_size for {symbol}: {e}")
        return DEFAULT_PARAMS['MIN_LOT']

# -------------------------
# POSITION MANAGEMENT
# -------------------------
def count_open_positions(symbol):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logging.error(f"Failed to get positions for {symbol}: {mt5.last_error()}")
            return 0
        count = sum(1 for pos in positions if pos.magic == 234000)
        logging.debug(f"Open positions for {symbol}: {count}")
        return count
    except Exception as e:
        logging.error(f"Error in count_open_positions for {symbol}: {e}")
        return 0

# -------------------------
# DATA SAVING FOR WEB APP
# -------------------------
def save_bot_data(symbol, analysis_data, account_info, open_trades):
    try:
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'analysis': {
                'predicted_price': safe_format(analysis_data.get('predicted_price'), '.5f'),
                'signal': analysis_data.get('signal', 'N/A'),
                'dynamic_threshold': safe_format(analysis_data.get('dynamic_threshold'), '.4%'),
                'confidence': safe_format(analysis_data.get('confidence'), '.2f'),
                'rsi': safe_format(analysis_data.get('rsi'), '.2f'),
                'atr': safe_format(analysis_data.get('atr'), '.5f'),
                'adx': safe_format(analysis_data.get('adx'), '.2f'),
                'vol': safe_format(analysis_data.get('vol'), '.6f'),
                'stoch_k': safe_format(analysis_data.get('stoch_k'), '.2f'),
                'stoch_d': safe_format(analysis_data.get('stoch_d'), '.2f'),
                'stoch_signal': analysis_data.get('stoch_signal', 'N/A')
            },
            'account': {
                'balance': account_info.balance if account_info else 0.0,
                'equity': account_info.equity if account_info else 0.0,
                'margin_free': account_info.margin_free if account_info else 0.0,
            },
            'open_trades': [
                {
                    'symbol': pos.symbol,
                    'ticket': pos.ticket,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'entry_price': pos.price_open,
                    'profit': pos.profit,
                } for pos in open_trades if pos.magic == 234000
            ]
        }
        with open('bot_data.json', 'a') as f:
            json.dump(data, f)
            f.write('\n')
    except Exception as e:
        logging.error(f"Error saving bot data for {symbol}: {e}")

# -------------------------
# MAIN TRADING LOOP
# -------------------------
def main():
    if not ensure_mt5_connection():
        return
    logging.info("Connected to MT5")
    print("Connected to MT5")

    for symbol in SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select {symbol}")
            print(f"Failed to select {symbol}")
            continue
        info = mt5.symbol_info(symbol)
        if not info:
            logging.error(f"No symbol info for {symbol}")
            print(f"No symbol info for {symbol}")
            continue
        tick = mt5.symbol_info_tick(symbol)
        market_active = tick and tick.time >= int(time.time()) - 60
        print(f"{symbol}: visible={info.visible}, trade_mode={info.trade_mode}, market_active={market_active}")
        logging.debug(f"{symbol}: visible={info.visible}, trade_mode={info.trade_mode}, market_active={market_active}")

        df = get_historical_data(symbol, mt5.TIMEFRAME_H1, DEFAULT_PARAMS['BARS_TO_FETCH'])
        if df is not None:
            RF_MODELS[symbol] = train_rf_model(symbol, df)
        else:
            RF_MODELS[symbol] = None
            logging.warning(f"Could not train RF model for {symbol} due to missing data")

    loop_count = 0
    last_retrain_time = datetime.now()
    try:
        while True:
            loop_count += 1
            logging.info(f"\n=== Loop {loop_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"\n=== Loop {loop_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

            if not ensure_mt5_connection():
                time.sleep(60)
                continue

            log_account_status()

            analysis_results = {}
            logging.info("Starting 1-hour market analysis phase")
            for symbol in SYMBOLS:
                df = get_historical_data(symbol, mt5.TIMEFRAME_H1, DEFAULT_PARAMS['ANALYSIS_BARS'])
                if df is None or len(df) < DEFAULT_PARAMS['ANALYSIS_BARS']:
                    logging.warning(f"Insufficient data for {symbol} analysis, skipping")
                    analysis_results[symbol] = (None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None)
                    continue
                pred_price, signal, dynamic_threshold, confidence, stoch_signal, adx, stoch_k, stoch_d = analyze_market_conditions(symbol, df)
                analysis_results[symbol] = (pred_price, signal, dynamic_threshold, confidence, stoch_signal, adx, stoch_k, stoch_d)

            account_info = mt5.account_info()
            if not account_info:
                logging.error("Failed to get account info")
                print("Failed to get account info")
                break
            if account_info.balance < DEFAULT_PARAMS['MIN_ACCOUNT_BALANCE']:
                logging.error(f"Balance too low: {account_info.balance}")
                print(f"Balance too low: {account_info.balance}")
                break

            total_positions = mt5.positions_total()
            if total_positions >= DEFAULT_PARAMS['MAX_OPEN_TRADES']:
                logging.info(f"Maximum total open trades ({DEFAULT_PARAMS['MAX_OPEN_TRADES']}) reached, skipping new trades")
                print(f"Maximum total open trades ({DEFAULT_PARAMS['MAX_OPEN_TRADES']}) reached, skipping new trades")
                for symbol in SYMBOLS:
                    manage_breakeven(symbol)
                    close_old_trades(symbol)
                    save_bot_data(symbol, {}, account_info, mt5.positions_get())
                time.sleep(DEFAULT_PARAMS['LOOP_SLEEP'])
                continue

            if (datetime.now() - last_retrain_time).total_seconds() >= 86400:
                for symbol in SYMBOLS:
                    df = get_historical_data(symbol, mt5.TIMEFRAME_H1, DEFAULT_PARAMS['BARS_TO_FETCH'])
                    if df is not None:
                        RF_MODELS[symbol] = train_rf_model(symbol, df)
                        logging.info(f"Retrained RF model for {symbol}")
                last_retrain_time = datetime.now()

            for symbol in SYMBOLS:
                manage_breakeven(symbol)
                close_old_trades(symbol)

            for symbol in SYMBOLS:
                try:
                    if not mt5.symbol_select(symbol, True):
                        logging.warning(f"Could not select {symbol}, skipping")
                        print(f"Could not select {symbol}, skipping")
                        continue
                    info = mt5.symbol_info(symbol)
                    if not info or info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                        logging.warning(f"{symbol} not tradable, skipping")
                        print(f"{symbol} not tradable, skipping")
                        continue
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick or tick.time < int(time.time()) - 60:
                        logging.warning(f"No recent tick data for {symbol}, market may be closed")
                        print(f"No recent tick data for {symbol}, market may be closed")
                        continue
                    logging.debug(f"{symbol} info: point={info.point}, trade_mode={info.trade_mode}")

                    open_positions = count_open_positions(symbol)
                    if open_positions >= DEFAULT_PARAMS['MAX_TRADES_PER_SYMBOL']:
                        logging.info(f"Maximum open trades ({DEFAULT_PARAMS['MAX_TRADES_PER_SYMBOL']}) for {symbol} reached, skipping new trades")
                        print(f"Maximum open trades ({DEFAULT_PARAMS['MAX_TRADES_PER_SYMBOL']}) for {symbol} reached, skipping new trades")
                        analysis = analysis_results.get(symbol, (None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None))
                        save_bot_data(symbol, {
                            'predicted_price': analysis[0],
                            'signal': analysis[1],
                            'dynamic_threshold': analysis[2],
                            'confidence': analysis[3],
                            'rsi': None,
                            'atr': None,
                            'adx': analysis[5],
                            'vol': None,
                            'stoch_k': analysis[6],
                            'stoch_d': analysis[7],
                            'stoch_signal': analysis[4]
                        }, account_info, mt5.positions_get())
                        continue

                    df = get_historical_data(symbol, mt5.TIMEFRAME_H1, DEFAULT_PARAMS['BARS_TO_FETCH'])
                    if df is None:
                        save_bot_data(symbol, {}, account_info, mt5.positions_get())
                        continue

                    analysis = analysis_results.get(symbol, (None, None, DEFAULT_PARAMS['DEFAULT_THRESHOLD'], None, None, None, None, None))
                    dynamic_threshold = analysis[2]
                    pred_price, rsi, vol, macd, bb_upper, bb_lower, atr, dynamic_threshold, confidence, signal, adx, stoch_k, stoch_d = predict_next_price(df, symbol, dynamic_threshold)
                    if pred_price is None or dynamic_threshold is None:
                        logging.warning(f"Prediction failed or invalid threshold for {symbol}, skipping")
                        print(f"Prediction failed or invalid threshold for {symbol}, skipping")
                        save_bot_data(symbol, {}, account_info, mt5.positions_get())
                        continue

                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        logging.warning(f"No tick data for {symbol}")
                        print(f"No tick data for {symbol}")
                        save_bot_data(symbol, {}, account_info, mt5.positions_get())
                        continue
                    ask, bid = tick.ask, tick.bid
                    point = info.point
                    logging.debug(f"{symbol} tick: ask={safe_format(ask, '.5f')}, bid={safe_format(bid, '.5f')}, point={point}")

                    lot = calculate_dynamic_lot_size(symbol, account_info.balance, atr, point)
                    pred_vs_ask = (pred_price - ask) / ask if pred_price is not None and ask != 0 else 0.0
                    pred_vs_bid = (pred_price - bid) / bid if pred_price is not None and bid != 0 else 0.0
                    stoch_signal = analysis[4]

                    logging.info(f"{symbol}: pred={safe_format(pred_price, '.5f')}, ask={safe_format(ask, '.5f')}, bid={safe_format(bid, '.5f')}, "
                                 f"Δask={safe_format(pred_vs_ask, '.4%')}, Δbid={safe_format(pred_vs_bid, '.4%')}, RSI={safe_format(rsi, '.2f')}, "
                                 f"ATR={safe_format(atr, '.5f')}, lot={safe_format(lot, '.2f')}, threshold={safe_format(dynamic_threshold, '.4%')}, "
                                 f"confidence={safe_format(confidence, '.2f')}, stoch_signal={stoch_signal}, open_positions={open_positions}")
                    print(f"{symbol}: pred={safe_format(pred_price, '.5f')}, ask={safe_format(ask, '.5f')}, bid={safe_format(bid, '.5f')}, "
                          f"Δask={safe_format(pred_vs_ask, '.4%')}, Δbid={safe_format(pred_vs_bid, '.4%')}, RSI={safe_format(rsi, '.2f')}, "
                          f"ATR={safe_format(atr, '.5f')}, lot={safe_format(lot, '.2f')}, threshold={safe_format(dynamic_threshold, '.4%')}, "
                          f"confidence={safe_format(confidence, '.2f')}, stoch_signal={stoch_signal}, open_positions={open_positions}")

                    tp_multiplier = SYMBOL_TP_MULTIPLIERS.get(symbol, DEFAULT_PARAMS['ATR_MULTIPLIER_TP'])
                    min_tp_distance = MIN_TP_DISTANCE.get(symbol, 0.0)

                    if (pred_price > ask and pred_vs_ask >= dynamic_threshold and
                            confidence >= DEFAULT_PARAMS['CONFIDENCE_THRESHOLD'] and signal == "BUY"):
                        tp_distance = max(atr * tp_multiplier, min_tp_distance) if atr is not None else min_tp_distance
                        tp_price = ask + tp_distance
                        ticket = place_order_no_sl(symbol, mt5.ORDER_TYPE_BUY, lot, ask, tp_price)
                        if ticket:
                            logging.info(f"{symbol} BUY placed at {safe_format(ask, '.5f')}, tp={safe_format(tp_price, '.5f')}, lot={safe_format(lot, '.2f')}, ticket={ticket}")
                            print(f"{symbol} BUY placed at {safe_format(ask, '.5f')}, tp={safe_format(tp_price, '.5f')}, lot={safe_format(lot, '.2f')}, ticket={ticket}")
                        else:
                            logging.error(f"{symbol} BUY failed: {mt5.last_error()}")
                            print(f"{symbol} BUY failed: {mt5.last_error()}")
                    elif (pred_price < bid and pred_vs_bid <= -dynamic_threshold and
                          confidence >= DEFAULT_PARAMS['CONFIDENCE_THRESHOLD'] and signal == "SELL"):
                        tp_distance = max(atr * tp_multiplier, min_tp_distance) if atr is not None else min_tp_distance
                        tp_price = bid - tp_distance
                        ticket = place_order_no_sl(symbol, mt5.ORDER_TYPE_SELL, lot, bid, tp_price)
                        if ticket:
                            logging.info(f"{symbol} SELL placed at {safe_format(bid, '.5f')}, tp={safe_format(tp_price, '.5f')}, lot={safe_format(lot, '.2f')}, ticket={ticket}")
                            print(f"{symbol} SELL placed at {safe_format(bid, '.5f')}, tp={safe_format(tp_price, '.5f')}, lot={safe_format(lot, '.2f')}, ticket={ticket}")
                        else:
                            logging.error(f"{symbol} SELL failed: {mt5.last_error()}")
                            print(f"{symbol} SELL failed: {mt5.last_error()}")
                    else:
                        logging.info(f"{symbol} no trade: prediction within threshold or low confidence "
                                     f"(Δask={safe_format(pred_vs_ask, '.4%')}, Δbid={safe_format(pred_vs_bid, '.4%')}, confidence={safe_format(confidence, '.2f')}, signal={signal})")
                        print(f"{symbol} no trade: prediction within threshold or low confidence")

                    save_bot_data(symbol, {
                        'predicted_price': pred_price,
                        'signal': signal,
                        'dynamic_threshold': dynamic_threshold,
                        'confidence': confidence,
                        'rsi': rsi,
                        'atr': atr,
                        'adx': adx,
                        'vol': vol,
                        'stoch_k': stoch_k,
                        'stoch_d': stoch_d,
                        'stoch_signal': stoch_signal
                    }, account_info, mt5.positions_get())

                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    print(f"Error processing {symbol}: {e}")
                    save_bot_data(symbol, {}, account_info, mt5.positions_get())

            logging.debug(f"Sleeping for {DEFAULT_PARAMS['LOOP_SLEEP']} seconds")
            time.sleep(DEFAULT_PARAMS['LOOP_SLEEP'])

    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown")
        print("MT5 shutdown")

if __name__ == "__main__":
    main()