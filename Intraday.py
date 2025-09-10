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

# import vectorized indicators
from indicators import add_basic_features, rsi as vec_rsi, atr as vec_atr, bollinger_bands as vec_bb, stochastic as vec_stoch, adx as vec_adx

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
# TECHNICAL INDICATOR WRAPPERS (call vectorized implementations)
# -------------------------
def calculate_rsi(df, period=14):
    try:
        series = vec_rsi(df['close'], period)
        rsi_val = series.iloc[-1] if len(series) > 0 else 50.0
        return float(rsi_val) if not pd.isna(rsi_val) else 50.0
    except Exception as e:
        logging.error(f"Error in calculate_rsi: {e}")
        return 50.0


def calculate_atr(df, period=14, symbol_point=0.0001):
    try:
        series = vec_atr(df, period)
        atr_val = series.iloc[-1] if len(series) > 0 else symbol_point
        return float(atr_val) if not pd.isna(atr_val) else symbol_point
    except Exception as e:
        logging.error(f"Error in calculate_atr: {e}")
        return symbol_point


def calculate_bollinger_bands(df, period=20, num_std=2):
    try:
        upper, middle, lower = vec_bb(df['close'], period=period, num_std=num_std)
        u = upper.iloc[-1] if len(upper) > 0 else df['close'].iloc[-1]
        l = lower.iloc[-1] if len(lower) > 0 else df['close'].iloc[-1]
        return (float(u) if not pd.isna(u) else df['close'].iloc[-1], float(l) if not pd.isna(l) else df['close'].iloc[-1])
    except Exception as e:
        logging.error(f"Error in calculate_bollinger_bands: {e}")
        close = df['close'].iloc[-1] if len(df) > 0 else 0.0
        return close, close


def calculate_stochastic(df, k_period=14, d_period=3):
    try:
        k_series, d_series = vec_stoch(df, k_period=k_period, d_period=d_period)
        k_val = k_series.iloc[-1] if len(k_series) > 0 else None
        d_val = d_series.iloc[-1] if len(d_series) > 0 else None
        if pd.isna(k_val) or pd.isna(d_val):
            return None, None
        return float(k_val), float(d_val)
    except Exception as e:
        logging.error(f"Error in calculate_stochastic: {e}")
        return None, None


def calculate_adx(df, period=14):
    try:
        series = vec_adx(df, period=period)
        adx_val = series.iloc[-1] if len(series) > 0 else 20.0
        return float(adx_val) if not pd.isna(adx_val) else 20.0
    except Exception as e:
        logging.error(f"Error in calculate_adx: {e}")
        return 20.0

# -------------------------
# MARKET ANALYSIS
# -------------------------
def analyze_market_conditions(symbol, df):
    try:
        logging.info(f"Starting market analysis for {symbol}")
        # compute vectorized indicators once
        df = df.copy()
        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        df = add_basic_features(df, rsi_period=DEFAULT_PARAMS['RSI_PERIOD'], atr_period=DEFAULT_PARAMS['ATR_PERIOD'])

        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else calculate_rsi(df, DEFAULT_PARAMS['RSI_PERIOD'])
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else calculate_atr(df, DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point)
        vol = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().iloc[-1]
        vol = 0.0 if pd.isna(vol) else vol
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else (df['close'].ewm(span=12).mean().iloc[-1] - df['close'].ewm(span=26).mean().iloc[-1])
        macd = 0.0 if pd.isna(macd) else macd
        bb_upper = df['bb_upper'].iloc[-1] if 'bb_upper' in df.columns else calculate_bollinger_bands(df)[0]
        bb_lower = df['bb_lower'].iloc[-1] if 'bb_lower' in df.columns else calculate_bollinger_bands(df)[1]
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else (calculate_stochastic(df)[0] if calculate_stochastic(df)[0] is not None else 50.0)
        stoch_d = df['stoch_d'].iloc[-1] if 'stoch_d' in df.columns else (calculate_stochastic(df)[1] if calculate_stochastic(df)[1] is not None else 50.0)
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else calculate_adx(df)
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
                     f"stoch_signal={stoch_signal}, dynamic_threshold={safe_format(dynamic_threshold, '.4%')}"
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
                time... (truncated)
        except Exception as e:
            logging.error(f"Error in train_rf_model for {symbol}: {e}")
            time.sleep(1)
            continue
    return None