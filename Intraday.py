import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

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
    'BARS_TO_FETCH': 1000,
    'LOOP_SLEEP': 600,
    'MIN_LOT': 0.01,
    'MAX_LOT': 1.0,
    'MAX_DRAWDOWN_PERCENT': 2.0,
    'MAX_OPEN_TRADES': 12,  # Updated to 12
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
    "BTCUSDm": 500.0,
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
MAX_TRADES_PER_SYMBOL = 2  # New limit

# -------------------------
# TECHNICAL INDICATORS (UNCHANGED)
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
        return rsi if not np.isnan(rsi) else None
    except Exception as e:
        logging.error(f"Error in calculate_rsi: {e}")
        return None

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
        return upper.iloc[-1], lower.iloc[-1]
    except Exception as e:
        logging.error(f"Error in calculate_bollinger_bands: {e}")
        return None, None

def calculate_stochastic(df, k_period=14, d_period=3):
    try:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        symbol = df['symbol'].iloc[0] if 'symbol' in df else 'unknown'
        logging.debug(f"Stochastic for {symbol}: %K={k.iloc[-1]:.2f}, %D={d.iloc[-1]:.2f}")
        return k.iloc[-1], d.iloc[-1]
    except Exception as e:
        logging.error(f"Error in calculate_stochastic: {e}")
        return None, None

# -------------------------
# ML MODEL TRAINING
# -------------------------
def train_rf_model(symbol, df):
    try:
        logging.info(f"Starting RF model training for {symbol}")
        df = df.copy()
        df['rsi'] = df['close'].diff().rolling(DEFAULT_PARAMS['RSI_PERIOD']).apply(
            lambda x: calculate_rsi(pd.DataFrame({'close': x}), DEFAULT_PARAMS['RSI_PERIOD'])
        )
        df['atr'] = df.apply(
            lambda x: calculate_atr(pd.DataFrame({
                'high': df['high'], 'low': df['low'], 'close': df['close']
            }), DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.0001),
            axis=1
        )
        df['vol'] = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
        df['lag1'] = df['close'].shift(1)
        df['lag2'] = df['close'].shift(2)
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
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

        features = ['rsi', 'atr', 'vol', 'macd', 'bb_upper', 'bb_lower', 'lag1', 'lag2', 'stoch_k', 'stoch_d', 'hour', 'day_of_week']
        df = df.dropna(subset=features + ['target'])

        if len(df) < 100:
            logging.warning(f"Insufficient data to train RF model for {symbol}: {len(df)} rows")
            return None

        X = df[features]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"RF model trained for {symbol}, test accuracy: {accuracy:.4f}, best params: {grid_search.best_params_}")

        return best_model
    except Exception as e:
        logging.error(f"Error training RF model for {symbol}: {e}")
        return None

# -------------------------
# PREDICTION LOGIC
# -------------------------
def predict_next_price(df, symbol):
    try:
        logging.debug(f"Starting prediction for {symbol}")
        rsi = calculate_rsi(df, DEFAULT_PARAMS['RSI_PERIOD'])
        atr = calculate_atr(df, DEFAULT_PARAMS['ATR_PERIOD'], mt5.symbol_info(symbol).point)
        vol = df['close'].pct_change().rolling(DEFAULT_PARAMS['VOLATILITY_PERIOD']).std().iloc[-1]
        macd = df['close'].ewm(span=12).mean().iloc[-1] - df['close'].ewm(span=26).mean().iloc[-1]
        bb_upper, bb_lower = calculate_bollinger_bands(df)
        lag1 = df['close'].iloc[-1]
        lag2 = df['close'].iloc[-2] if len(df) > 1 else lag1
        stoch_k, stoch_d = calculate_stochastic(df)
        hour = df['time'].iloc[-1].hour
        day_of_week = df['time'].iloc[-1].dayofweek

        if symbol not in RF_MODELS or RF_MODELS[symbol] is None:
            logging.warning(f"No RF model for {symbol}, falling back to MA crossover")
            short_ma = df['close'].rolling(20).mean().iloc[-1]
            long_ma = df['close'].rolling(100).mean().iloc[-1]
            if short_ma > long_ma:
                pred_price = df['close'].iloc[-1] * 1.01
                logging.info(f"{symbol} predicted BUY (MA): pred_price={pred_price:.5f}")
            else:
                pred_price = df['close'].iloc[-1] * 0.99
                logging.info(f"{symbol} predicted SELL (MA): pred_price={pred_price:.5f}")
            return pred_price, rsi, vol, macd, bb_upper, bb_lower, atr

        features = pd.DataFrame({
            'rsi': [rsi],
            'atr': [atr],
            'vol': [vol],
            'macd': [macd],
            'bb_upper': [bb_upper],
            'bb_lower': [bb_lower],
            'lag1': [lag1],
            'lag2': [lag2],
            'stoch_k': [stoch_k],
            'stoch_d': [stoch_d],
            'hour': [hour],
            'day_of_week': [day_of_week]
        })

        model = RF_MODELS[symbol]
        pred_class = model.predict(features)[0]
        current_price = df['close'].iloc[-1]

        if pred_class == 1:
            pred_price = current_price * 1.01
            logging.info(f"{symbol} predicted BUY (RF): pred_price={pred_price:.5f}, class={pred_class}")
        elif pred_class == -1:
            pred_price = current_price * 0.99
            logging.info(f"{symbol} predicted SELL (RF): pred_price={pred_price:.5f}, class={pred_class}")
        else:
            pred_price = current_price
            logging.info(f"{symbol} predicted NO TRADE (RF): pred_price={pred_price:.5f}, class={pred_class}")

        return pred_price, rsi, vol, macd, bb_upper, bb_lower, atr
    except Exception as e:
        logging.error(f"Error in predict_next_price for {symbol}: {e}")
        return None, None, None, None, None, None, None

# -------------------------
# DATA FETCHING
# -------------------------
def get_historical_data(symbol, timeframe, bars=1000):
    try:
        logging.debug(f"Fetching historical data for {symbol}, timeframe={timeframe}, bars={bars}")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logging.warning(f"No historical data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol
        logging.debug(f"Historical data fetched for {symbol}: {len(df)} bars")
        return df
    except Exception as e:
        logging.error(f"Error in get_historical_data for {symbol}: {e}")
        return None

# -------------------------
# ORDER MANAGEMENT (UNCHANGED)
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
# BREAKEVEN MANAGEMENT
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
        stops_level = info.stops_level * info.point

        # Log open positions
        for pos in positions:
            if pos.magic == 234000:
                logging.info(f"Open position for {symbol}: ticket={pos.ticket}, type={'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}, "
                             f"entry_price={pos.price_open:.5f}, current_price={pos.price_current:.5f}, profit={pos.profit:.2f}, "
                             f"sl={pos.sl:.5f}, tp={pos.tp:.5f}")

        for pos in positions:
            if pos.magic != 234000:
                continue
            if pos.sl != 0.0:
                continue

            profit = pos.profit
            entry_price = pos.price_open
            current_price = pos.price_current
            order_type = pos.type

            if profit >= profit_threshold:
                secure_profit = profit * secure_percent
                if order_type == mt5.ORDER_TYPE_BUY:
                    contract_size = info.trade_contract_size or 100000
                    price_adjustment = secure_profit / (pos.volume * contract_size)
                    breakeven_sl = entry_price + price_adjustment
                    if current_price - breakeven_sl < stops_level:
                        breakeven_sl = current_price - stops_level
                else:
                    contract_size = info.trade_contract_size or 100000
                    price_adjustment = secure_profit / (pos.volume * contract_size)
                    breakeven_sl = entry_price - price_adjustment
                    if breakeven_sl - current_price < stops_level:
                        breakeven_sl = current_price + stops_level

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": symbol,
                    "sl": float(breakeven_sl),
                    "tp": float(pos.tp),
                }
                logging.debug(f"Breakeven request for {symbol} ticket={pos.ticket}: sl={breakeven_sl}, profit={profit}")
                res = mt5.order_send(request)
                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to set breakeven for {symbol} ticket={pos.ticket}: {mt5.last_error()}")
                else:
                    logging.info(f"Breakeven set for {symbol} ticket={pos.ticket}: sl={breakeven_sl}, secured_profit={secure_profit:.2f}")
    except Exception as e:
        logging.error(f"Error in manage_breakeven for {symbol}: {e}")

# -------------------------
# LOT SIZE CALCULATION (UNCHANGED)
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
    """Count the number of open positions for a given symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logging.error(f"Failed to get positions for {symbol}: {mt5.last_error()}")
            return 0
        count = sum(1 for pos in positions if pos.magic == 234000)  # Count only bot's trades
        logging.debug(f"Open positions for {symbol}: {count}")
        return count
    except Exception as e:
        logging.error(f"Error in count_open_positions for {symbol}: {e}")
        return 0

# -------------------------
# MAIN TRADING LOOP
# -------------------------
def main():
    logging.debug("Attempting MT5 initialization")
    if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        print(f"MT5 initialization failed: {mt5.last_error()}")
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
    try:
        while True:
            loop_count += 1
            logging.info(f"\n=== Loop {loop_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"\n=== Loop {loop_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

            account_info = mt5.account_info()
            if not account_info:
                logging.error("Failed to get account info")
                print("Failed to get account info")
                break
            if account_info.balance < DEFAULT_PARAMS['MIN_ACCOUNT_BALANCE']:
                logging.error(f"Balance too low: {account_info.balance}")
                print(f"Balance too low: {account_info.balance}")
                break
            logging.debug(f"Account balance: {account_info.balance}, equity: {account_info.equity}, margin_free={account_info.margin_free}")

            total_positions = mt5.positions_total()
            if total_positions >= DEFAULT_PARAMS['MAX_OPEN_TRADES']:
                logging.info(f"Maximum total open trades ({DEFAULT_PARAMS['MAX_OPEN_TRADES']}) reached, skipping new trades")
                print(f"Maximum total open trades ({DEFAULT_PARAMS['MAX_OPEN_TRADES']}) reached, skipping new trades")
                for symbol in SYMBOLS:
                    manage_breakeven(symbol)
                time.sleep(DEFAULT_PARAMS['LOOP_SLEEP'])
                continue

            for symbol in SYMBOLS:
                manage_breakeven(symbol)

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

                    # Check per-symbol trade limit
                    open_positions = count_open_positions(symbol)
                    if open_positions >= MAX_TRADES_PER_SYMBOL:
                        logging.info(f"Maximum open trades ({MAX_TRADES_PER_SYMBOL}) for {symbol} reached, skipping new trades")
                        print(f"Maximum open trades ({MAX_TRADES_PER_SYMBOL}) for {symbol} reached, skipping new trades")
                        continue

                    df = get_historical_data(symbol, mt5.TIMEFRAME_H1, DEFAULT_PARAMS['BARS_TO_FETCH'])
                    if df is None:
                        continue

                    pred_price, rsi, vol, macd, bb_upper, bb_lower, atr = predict_next_price(df, symbol)
                    if pred_price is None:
                        logging.warning(f"Prediction failed for {symbol}, skipping")
                        print(f"Prediction failed for {symbol}, skipping")
                        continue

                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        logging.warning(f"No tick data for {symbol}")
                        print(f"No tick data for {symbol}")
                        continue
                    ask, bid = tick.ask, tick.bid
                    point = info.point
                    logging.debug(f"{symbol} tick: ask={ask:.5f}, bid={bid:.5f}, point={point}")

                    lot = calculate_dynamic_lot_size(symbol, account_info.balance, atr, point)
                    pct_threshold = max(0.0001, atr / df['close'].iloc[-1] * 0.5)
                    pred_vs_ask = (pred_price - ask) / ask
                    pred_vs_bid = (pred_price - bid) / bid

                    logging.info(f"{symbol}: pred={pred_price:.5f}, ask={ask:.5f}, bid={bid:.5f}, "
                                 f"Δask={pred_vs_ask:.4%}, Δbid={pred_vs_bid:.4%}, RSI={rsi:.2f}, ATR={atr:.5f}, lot={lot}, threshold={pct_threshold:.4%}, open_positions={open_positions}")
                    print(f"{symbol}: pred={pred_price:.5f}, ask={ask:.5f}, bid={bid:.5f}, "
                          f"Δask={pred_vs_ask:.4%}, Δbid={pred_vs_bid:.4%}, RSI={rsi:.2f}, ATR={atr:.5f}, lot={lot}, threshold={pct_threshold:.4%}, open_positions={open_positions}")

                    tp_multiplier = SYMBOL_TP_MULTIPLIERS.get(symbol, DEFAULT_PARAMS['ATR_MULTIPLIER_TP'])
                    min_tp_distance = MIN_TP_DISTANCE.get(symbol, 0.0)

                    if pred_price > ask and pred_vs_ask >= pct_threshold:
                        tp_distance = max(atr * tp_multiplier, min_tp_distance)
                        tp_price = ask + tp_distance
                        ticket = place_order_no_sl(symbol, mt5.ORDER_TYPE_BUY, lot, ask, tp_price)
                        if ticket:
                            logging.info(f"{symbol} BUY placed at {ask}, tp={tp_price}, lot={lot}, ticket={ticket}")
                            print(f"{symbol} BUY placed at {ask}, tp={tp_price}, lot={lot}, ticket={ticket}")
                        else:
                            logging.error(f"{symbol} BUY failed")
                            print(f"{symbol} BUY failed")
                    elif pred_price < bid and pred_vs_bid <= -pct_threshold:
                        tp_distance = max(atr * tp_multiplier, min_tp_distance)
                        tp_price = bid - tp_distance
                        ticket = place_order_no_sl(symbol, mt5.ORDER_TYPE_SELL, lot, bid, tp_price)
                        if ticket:
                            logging.info(f"{symbol} SELL placed at {bid}, tp={tp_price}, lot={lot}, ticket={ticket}")
                            print(f"{symbol} SELL placed at {bid}, tp={tp_price}, lot={lot}, ticket={ticket}")
                        else:
                            logging.error(f"{symbol} SELL failed")
                            print(f"{symbol} SELL failed")
                    else:
                        logging.info(f"{symbol} no trade: prediction within threshold (Δask={pred_vs_ask:.4%}, Δbid={pred_vs_bid:.4%})")
                        print(f"{symbol} no trade: prediction within threshold")

                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    print(f"Error processing {symbol}: {e}")

            logging.debug(f"Sleeping for {DEFAULT_PARAMS['LOOP_SLEEP']} seconds")
            time.sleep(DEFAULT_PARAMS['LOOP_SLEEP'])

    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown")
        print("MT5 shutdown")

if __name__ == "__main__":
    main()
