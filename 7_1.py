import ccxt
import openai
import pandas as pd
import talib as ta
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from prophet import Prophet
import time
import os

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from pmdarima import auto_arima
import joblib
from xgboost import XGBRegressor

# Отключение предупреждений TensorFlow
tf.get_logger().setLevel('ERROR')

# Создание подкласса CustomXGBRegressor для корректной работы с scikit-learn
class CustomXGBRegressor(XGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __sklearn_tags__(self):
        return {'requires_y': True}

# Настройки API KuCoin

# Настройки API OpenAI

# Инициализация клиента KuCoin
exchange = ccxt.kucoin({
    'apiKey': KUCOIN_API_KEY,
    'secret': KUCOIN_API_SECRET,
    'password': KUCOIN_API_PASSPHRASE,
})

# Настройка логирования с использованием utf-8, запись в файл и вывод в консоль
log_file_path = "D:\\7\\script_logs.txt"  # Укажите желаемый путь к файлу логов

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Создание файлового обработчика с кодировкой utf-8
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Создание консольного обработчика
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Создание форматтера и добавление его к обработчикам
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Удаление всех существующих обработчиков и добавление новых
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_model_dir(symbol):
    symbol_dir = symbol.replace("/", "_")
    model_dir = os.path.join("models", symbol_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def fetch_ohlcv(symbol, timeframe, limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            logger.warning(f"No OHLCV data returned for {symbol} ({timeframe}).")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')  # Установлено utc=True

        if df["timestamp"].isnull().any():
            logger.warning("Некоторые значения времени не смогли быть преобразованы в datetime.")

        logger.info(f"Типы данных после загрузки для {symbol} ({timeframe}):\n{df.dtypes}")

        return df
    except Exception as e:
        logger.error(f"Ошибка при запросе данных OHLCV для {symbol} ({timeframe}): {e}")
        return None

def analyze_spikes(df):
    if df is None or df.empty or len(df) < 2:
        return df
    df['pct_change'] = df['close'].pct_change() * 100
    df['spike'] = abs(df['pct_change']) > 5
    return df

def apply_candlestick_patterns(df):
    if df is None or df.empty or len(df) < 20:
        return df

    df['Hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['Bullish_Engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) > 0
    df['Bearish_Engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) < 0
    df['Doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['Morning_Star'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['Evening_Star'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['Shooting_Star'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['Three_White_Soldiers'] = ta.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
    df['Three_Black_Crows'] = ta.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])

    return df

def apply_technical_indicators(df):
    if df is None or df.empty or len(df) < 20:
        return df

    if not pd.api.types.is_datetime64_ns_dtype(df['timestamp']):
        logger.error("Столбец timestamp не является datetime.")
        return df

    df['RSI'] = ta.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'],
                                        fastk_period=14, slowk_period=3, slowk_matype=0,
                                        slowd_period=3, slowd_matype=0)
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    df = analyze_spikes(df)
    df = apply_candlestick_patterns(df)

    return df

def forecast_with_prophet(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"prophet_model_{timeframe}.pkl")
    try:
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"Недостаточно данных для прогноза с Prophet для {timeframe}.")
            return None

        df_prophet = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_localize(None)  # Удалено timezone

        # Сортировка данных по времени
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = joblib.load(f)
            logger.info(f"Загружена сохранённая модель Prophet для {timeframe}.")
        else:
            model = Prophet()
            model.fit(df_prophet)
            with open(model_path, "wb") as f:
                joblib.dump(model, f)
            logger.info(f"Обучена и сохранена новая модель Prophet для {timeframe}.")

        # Определение горизонта прогнозирования в зависимости от таймфрейма
        if timeframe == "1h":
            periods = 24  # Прогноз на 24 часа вперед
            freq = 'H'
            delta = timedelta(hours=1)
        elif timeframe == "1d":
            periods = 1  # Прогноз на 1 день вперед
            freq = 'D'
            delta = timedelta(days=1)
        else:
            periods = 24
            freq = 'H'
            delta = timedelta(hours=1)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        compact_forecast = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].tail(periods)
        compact_forecast.columns = ['Время', f'Прогноз_Prophet_{timeframe}', f'Прогноз_Prophet_Upper_{timeframe}', f'Прогноз_Prophet_Lower_{timeframe}']
        logger.info(f"Прогноз с Prophet для {timeframe} успешно создан.")
        return compact_forecast
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с Prophet для {timeframe}: {e}")
        return None

def create_features(df, look_back=30, additional_lags=5):
    if df is None or df.empty:
        return df

    if not pd.api.types.is_datetime64_ns_dtype(df['timestamp']):
        logger.error("Столбец timestamp не datetime.")
        return df

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    df['volume_ma14'] = df['volume'].rolling(14).mean().bfill()
    df['vol_ratio'] = df['volume'] / df['volume_ma14']

    for lag in range(1, additional_lags+1):
        df[f'lag_{lag}'] = df['close'].shift(lag)

    df = df.dropna()
    return df

def prepare_xy(df, look_back=30):
    if df is None or df.empty:
        return None, None

    prices = df['close'].values
    features_list = ['RSI', 'MACD_hist', 'ATR', 'slowk', 'slowd', 'day_of_week', 'hour', 'vol_ratio']
    lag_cols = [col for col in df.columns if col.startswith('lag_')]
    features_list.extend(lag_cols)

    for col in features_list:
        if col not in df.columns:
            df[col] = 0.0

    feature_data = df[features_list].values

    X, Y = [], []
    full_prices = df['close'].values
    for i in range(look_back, len(df)-1):
        past_prices = full_prices[i-look_back:i]
        other_features = feature_data[i]
        x_row = np.concatenate([past_prices, other_features])
        X.append(x_row)
        Y.append(full_prices[i+1])

    if len(X) == 0:
        return None, None

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def optimize_lightgbm(X_train, Y_train):
    param_grid = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500]
    }
    lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', n_jobs=1)  # Ограничение потоков
    grid = GridSearchCV(lgbm, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
    grid.fit(X_train, Y_train)
    logger.info(f"Лучшие параметры LightGBM: {grid.best_params_}")
    return grid.best_estimator_

def optimize_random_forest(X_train, Y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=1)  # Ограничение потоков
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
    grid.fit(X_train, Y_train)
    logger.info(f"Лучшие параметры Random Forest: {grid.best_params_}")
    return grid.best_estimator_

def optimize_catboost(X_train, Y_train):
    param_grid = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [200, 500, 1000]
    }
    cat = CatBoostRegressor(loss_function='RMSE', verbose=False, thread_count=1)  # Ограничение потоков
    grid = GridSearchCV(cat, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
    grid.fit(X_train, Y_train)
    logger.info(f"Лучшие параметры CatBoost: {grid.best_params_}")
    return grid.best_estimator_

def optimize_xgboost(X_train, Y_train):
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500]
    }
    xgb = CustomXGBRegressor(objective='reg:squarederror', random_state=42, use_label_encoder=False, eval_metric='rmse', n_jobs=1)  # Ограничение потоков
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
    try:
        grid.fit(X_train, Y_train)
        logger.info(f"Лучшие параметры XGBoost: {grid.best_params_}")
        return grid.best_estimator_
    except AttributeError as e:
        logger.error(f"Ошибка при оптимизации XGBoost: {e}")
        return None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при оптимизации XGBoost: {e}")
        return None

def forecast_with_lightgbm(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"lightgbm_model_{timeframe}.pkl")
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с LightGBM для {timeframe}.")
            return None

        df = create_features(df)
        X, Y = prepare_xy(df)
        if X is None or Y is None or len(X) < 2:
            logger.warning(f"Недостаточно данных для прогноза с LightGBM для {timeframe} после подготовки.")
            return None

        X_train, Y_train = X[:-1], Y[:-1]
        X_forecast = X[-1].reshape(1, -1)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Загружена сохранённая модель LightGBM для {timeframe}.")
        else:
            # Оптимизация гиперпараметров LightGBM
            optimized_model = optimize_lightgbm(X_train, Y_train)
            joblib.dump(optimized_model, model_path)
            model = optimized_model
            logger.info(f"Обучена и сохранена новая модель LightGBM для {timeframe}.")

        pred = model.predict(X_forecast)

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Определение временного шага
        if timeframe == "1h":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        elif timeframe == "1d":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(days=1)
        else:
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)

        # Расчет прогнозов максимума и минимума
        forecast_max = pred.flatten()[0] + atr
        forecast_min = pred.flatten()[0] - atr

        df_lightgbm_forecast = pd.DataFrame({
            'Время': [forecast_time],
            f'Прогноз_LightGBM_{timeframe}': [pred.flatten()[0]],
            f'Прогноз_LightGBM_Upper_{timeframe}': [forecast_max],
            f'Прогноз_LightGBM_Lower_{timeframe}': [forecast_min]
        })
        logger.info(f"Прогноз с LightGBM для {timeframe} успешно создан.")
        return df_lightgbm_forecast

    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с LightGBM для {timeframe}: {e}")
        return None

def forecast_with_random_forest(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"random_forest_model_{timeframe}.pkl")
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с Random Forest для {timeframe}.")
            return None

        df = create_features(df)
        X, Y = prepare_xy(df)
        if X is None or Y is None or len(X) < 2:
            logger.warning(f"Недостаточно данных для прогноза с Random Forest для {timeframe} после подготовки.")
            return None

        X_train, Y_train = X[:-1], Y[:-1]
        X_forecast = X[-1].reshape(1, -1)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Загружена сохранённая модель Random Forest для {timeframe}.")
        else:
            # Оптимизация гиперпараметров Random Forest
            optimized_model = optimize_random_forest(X_train, Y_train)
            joblib.dump(optimized_model, model_path)
            model = optimized_model
            logger.info(f"Обучена и сохранена новая модель Random Forest для {timeframe}.")

        pred = model.predict(X_forecast)

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Определение временного шага
        if timeframe == "1h":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        elif timeframe == "1d":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(days=1)
        else:
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)

        # Расчет прогнозов максимума и минимума
        forecast_max = pred.flatten()[0] + atr
        forecast_min = pred.flatten()[0] - atr

        df_rf_forecast = pd.DataFrame({
            'Время': [forecast_time],
            f'Прогноз_RF_{timeframe}': [pred.flatten()[0]],
            f'Прогноз_RF_Upper_{timeframe}': [forecast_max],
            f'Прогноз_RF_Lower_{timeframe}': [forecast_min]
        })
        logger.info(f"Прогноз с Random Forest для {timeframe} успешно создан.")
        return df_rf_forecast

    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с Random Forest для {timeframe}: {e}")
        return None

def forecast_with_catboost(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"catboost_model_{timeframe}.cbm")
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с CatBoost для {timeframe}.")
            return None

        df = create_features(df)
        X, Y = prepare_xy(df)
        if X is None or Y is None or len(X) < 2:
            logger.warning(f"Недостаточно данных для прогноза с CatBoost для {timeframe} после подготовки.")
            return None

        # Масштабирование целевой переменной
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

        X_train, Y_train = X[:-1], Y_scaled[:-1]
        X_forecast = X[-1].reshape(1, -1)

        if os.path.exists(model_path):
            model = CatBoostRegressor()
            model.load_model(model_path)
            logger.info(f"Загружена сохранённая модель CatBoost для {timeframe}.")
        else:
            # Оптимизация гиперпараметров CatBoost
            optimized_model = optimize_catboost(X_train, Y_train)
            model = optimized_model
            model.save_model(model_path)
            logger.info(f"Обучена и сохранена новая модель CatBoost для {timeframe}.")

        pred_scaled = model.predict(X_forecast)
        # Обратное масштабирование предсказания
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Определение временного шага
        if timeframe == "1h":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        elif timeframe == "1d":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(days=1)
        else:
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)

        # Расчет прогнозов максимума и минимума
        forecast_max = pred.flatten()[0] + atr
        forecast_min = pred.flatten()[0] - atr

        df_catboost_forecast = pd.DataFrame({
            'Время': [forecast_time],
            f'Прогноз_CatBoost_{timeframe}': [pred.flatten()[0]],
            f'Прогноз_CatBoost_Upper_{timeframe}': [forecast_max],
            f'Прогноз_CatBoost_Lower_{timeframe}': [forecast_min]
        })
        logger.info(f"Прогноз с CatBoost для {timeframe} успешно создан.")
        return df_catboost_forecast
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с CatBoost для {timeframe}: {e}")
        return None

def forecast_with_xgboost(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"xgboost_model_{timeframe}.pkl")  # Используем .pkl для joblib
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с XGBoost для {timeframe}.")
            return None

        df = create_features(df)
        X, Y = prepare_xy(df)
        if X is None or Y is None or len(X) < 2:
            logger.warning(f"Недостаточно данных для прогноза с XGBoost для {timeframe} после подготовки.")
            return None

        # Масштабирование целевой переменной
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

        X_train, Y_train = X[:-1], Y_scaled[:-1]
        X_forecast = X[-1].reshape(1, -1)

        if os.path.exists(model_path):
            model = joblib.load(model_path)  # Используем joblib для загрузки
            logger.info(f"Загружена сохранённая модель XGBoost для {timeframe}.")
        else:
            # Оптимизация гиперпараметров XGBoost
            optimized_model = optimize_xgboost(X_train, Y_train)
            if optimized_model is not None:
                joblib.dump(optimized_model, model_path)  # Используем joblib для сохранения
                model = optimized_model
                logger.info(f"Обучена и сохранена новая модель XGBoost для {timeframe}.")
            else:
                logger.warning(f"Оптимизированная модель XGBoost для {timeframe} не была создана.")
                return None

        pred_scaled = model.predict(X_forecast)
        # Обратное масштабирование предсказания
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Определение временного шага
        if timeframe == "1h":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        elif timeframe == "1d":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(days=1)
        else:
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)

        # Расчет прогнозов максимума и минимума
        forecast_max = pred.flatten()[0] + atr
        forecast_min = pred.flatten()[0] - atr

        df_xgb_forecast = pd.DataFrame({
            'Время': [forecast_time],
            f'Прогноз_XGBoost_{timeframe}': [pred.flatten()[0]],
            f'Прогноз_XGBoost_Upper_{timeframe}': [forecast_max],
            f'Прогноз_XGBoost_Lower_{timeframe}': [forecast_min]
        })
        logger.info(f"Прогноз с XGBoost для {timeframe} успешно создан.")
        return df_xgb_forecast
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с XGBoost для {timeframe}: {e}")
        return None

def forecast_with_lstm(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"lstm_model_{timeframe}.h5")
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с LSTM для {timeframe}.")
            return None

        df = create_features(df)
        X, Y = prepare_xy(df)
        if X is None or Y is None or len(X) < 2:
            logger.warning(f"Недостаточно данных для прогноза с LSTM для {timeframe} после подготовки.")
            return None

        # Масштабирование признаков и целевой переменной
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler_X.fit_transform(X)
        Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

        X_train, Y_train = X_scaled[:-1], Y_scaled[:-1]
        X_forecast = X_scaled[-1].reshape(1, -1)

        look_back = 30  # Используйте тот же look_back, что и в prepare_xy
        if X_train.shape[1] < look_back:
            logger.error(f"Недостаточно признаков для формирования look_back={look_back}")
            return None

        X_train_prices = X_train[:, :look_back].reshape(-1, look_back, 1)
        X_forecast_prices = X_forecast[:, :look_back].reshape(-1, look_back, 1)

        if os.path.exists(model_path):
            model = load_model(model_path)
            logger.info(f"Загружена сохранённая модель LSTM для {timeframe}.")
            # Компилируем модель для устранения предупреждений
            model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            model = Sequential()
            model.add(LSTM(50, input_shape=(look_back, 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            # Добавление EarlyStopping для предотвращения переобучения
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            logger.info(f"Начало обучения модели LSTM для {timeframe}.")
            model.fit(X_train_prices, Y_train, epochs=10, batch_size=16, verbose=0, validation_split=0.2, callbacks=[early_stop])  # Уменьшено количество эпох
            model.save(model_path)
            logger.info(f"Обучена и сохранена новая модель LSTM для {timeframe}.")

        lstm_pred_scaled = model.predict(X_forecast_prices)
        # Обратное масштабирование предсказания
        pred_price = scaler_y.inverse_transform(lstm_pred_scaled).flatten()

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Определение временного шага
        if timeframe == "1h":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        elif timeframe == "1d":
            forecast_time = df['timestamp'].iloc[-1] + timedelta(days=1)
        else:
            forecast_time = df['timestamp'].iloc[-1] + timedelta(hours=1)

        # Расчет прогнозов максимума и минимума
        forecast_max = pred_price[0] + atr
        forecast_min = pred_price[0] - atr

        df_lstm_forecast = pd.DataFrame({
            'Время': [forecast_time],
            f'Прогноз_LSTM_{timeframe}': [pred_price[0]],
            f'Прогноз_LSTM_Upper_{timeframe}': [forecast_max],
            f'Прогноз_LSTM_Lower_{timeframe}': [forecast_min]
        })
        logger.info(f"Прогноз с LSTM для {timeframe} успешно создан.")
        return df_lstm_forecast
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с LSTM для {timeframe}: {e}")
        return None

def forecast_with_sarima(df, symbol, timeframe):
    model_dir = get_model_dir(symbol)
    model_path = os.path.join(model_dir, f"sarima_model_{timeframe}.pkl")
    try:
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Недостаточно данных для прогноза с SARIMA для {timeframe}.")
            return None

        y = df['close']

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Загружена сохранённая модель SARIMA для {timeframe}.")
        else:
            model = auto_arima(y, seasonal=False, error_action='ignore', suppress_warnings=True)
            joblib.dump(model, model_path)
            logger.info(f"Обучена и сохранена новая модель SARIMA для {timeframe}.")

        # Определение горизонта прогнозирования в зависимости от таймфрейма
        if timeframe == "1h":
            periods = 24  # Прогноз на 24 часа вперед
        elif timeframe == "1d":
            periods = 1  # Прогноз на 1 день вперед
        else:
            periods = 24

        forecast = model.predict(n_periods=periods)

        # Создание временных меток для прогноза
        last_time = df['timestamp'].iloc[-1]
        forecast_times = []
        for i in range(1, periods + 1):
            if timeframe == "1h":
                forecast_times.append(last_time + timedelta(hours=i))
            elif timeframe == "1d":
                forecast_times.append(last_time + timedelta(days=i))
            else:
                forecast_times.append(last_time + timedelta(hours=i))

        # Получение ATR для определения диапазона прогноза
        atr = df['ATR'].iloc[-1]

        # Расчет прогнозов максимума и минимума
        forecast_max = forecast + atr
        forecast_min = forecast - atr

        df_sarima_forecast = pd.DataFrame({
            'Время': forecast_times,
            f'Прогноз_SARIMA_{timeframe}': forecast,
            f'Прогноз_SARIMA_Upper_{timeframe}': forecast_max,
            f'Прогноз_SARIMA_Lower_{timeframe}': forecast_min
        })
        logger.info(f"Прогноз с SARIMA для {timeframe} успешно создан.")
        return df_sarima_forecast
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании с SARIMA для {timeframe}: {e}")
        return None

def combine_forecasts_max(all_forecasts_max):
    dfs = []
    for model, df_model in all_forecasts_max.items():
        if df_model is not None and not df_model.empty:
            try:
                pred_col = [col for col in df_model.columns if col.startswith(f'Прогноз_{model}_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
                df_temp = df_model[['Время', pred_col]].rename(columns={pred_col: 'pred_max'})
                df_temp['model'] = model
                dfs.append(df_temp)
            except IndexError:
                logger.warning(f"Прогноз для максимума не найден для модели {model} и таймфрейма.")
    if not dfs:
        return None
    combined = pd.concat(dfs)
    combined['Время'] = pd.to_datetime(combined['Время'], utc=True)
    grouped_max = combined.groupby(['Время']).agg({'pred_max': 'mean'}).reset_index()
    grouped_max.columns = ['Время', 'Прогноз_Ансамбль_Max']
    return grouped_max

def combine_forecasts_min(all_forecasts_min):
    dfs = []
    for model, df_model in all_forecasts_min.items():
        if df_model is not None and not df_model.empty:
            try:
                pred_col = [col for col in df_model.columns if col.startswith(f'Прогноз_{model}_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
                df_temp = df_model[['Время', pred_col]].rename(columns={pred_col: 'pred_min'})
                df_temp['model'] = model
                dfs.append(df_temp)
            except IndexError:
                logger.warning(f"Прогноз для минимума не найден для модели {model} и таймфрейма.")
    if not dfs:
        return None
    combined = pd.concat(dfs)
    combined['Время'] = pd.to_datetime(combined['Время'], utc=True)
    grouped_min = combined.groupby(['Время']).agg({'pred_min': 'mean'}).reset_index()
    grouped_min.columns = ['Время', 'Прогноз_Ансамбль_Min']
    return grouped_min

def combine_forecasts(all_forecasts):
    dfs = []
    for model, df_model in all_forecasts.items():
        if df_model is not None and not df_model.empty:
            # Стандартизируем название столбца прогноза
            pred_col = None
            if any(col.startswith('Прогноз_Prophet_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_Prophet_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_LightGBM_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_LightGBM_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_CatBoost_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_CatBoost_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_RF_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_RF_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_LSTM_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_LSTM_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_SARIMA_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_SARIMA_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            elif any(col.startswith('Прогноз_XGBoost_') and not col.endswith('Upper') and not col.endswith('Lower') for col in df_model.columns):
                pred_col = [col for col in df_model.columns if col.startswith('Прогноз_XGBoost_') and not col.endswith('Upper') and not col.endswith('Lower')][0]
            else:
                logger.warning(f"Неизвестный столбец прогноза для модели {model}.")
                continue

            df_temp = df_model[['Время', pred_col]].rename(columns={pred_col: 'pred'})
            df_temp['model'] = model
            dfs.append(df_temp)

    if not dfs:
        return None

    combined = pd.concat(dfs)
    # Убедимся, что все временные метки имеют временную зону UTC
    combined['Время'] = pd.to_datetime(combined['Время'], utc=True)
    grouped = combined.groupby(['Время']).agg({'pred': 'mean'}).reset_index()
    grouped.columns = ['Время', 'Прогноз_Ансамбль']
    return grouped

def analyze_with_openai(
    data,
    symbol,
    forecasts_max,
    forecasts_lightgbm_max,
    forecasts_catboost_max,
    forecasts_random_forest_max,
    forecasts_lstm_max=None,
    forecasts_sarima_max=None,
    forecasts_xgboost_max=None,
    forecasts_min=None,
    forecasts_lightgbm_min=None,
    forecasts_catboost_min=None,
    forecasts_random_forest_min=None,
    forecasts_lstm_min=None,
    forecasts_sarima_min=None,
    forecasts_xgboost_min=None,
    current_price=None,
    max_forecast_price=None,
    min_forecast_price=None,
    realistic_percentage=None,
    forecast_dates=None,
    model_statuses=None
):
    analyses = []

    for timeframe in data.keys():
        if timeframe not in current_price or timeframe not in forecast_dates:
            logger.warning(f"Недостаточно данных для анализа таймфрейма {timeframe}.")
            continue

        # Информация о текущей цене
        if current_price[timeframe] is not None:
            price_info = f"Текущая цена: {current_price[timeframe]:.4f}"
        else:
            price_info = "Текущая цена: Не доступна"

        # Информация о дате прогноза
        if forecast_dates[timeframe] is not None:
            forecast_date_info = f"Прогнозные данные до: {forecast_dates[timeframe].strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            forecast_date_info = "Прогнозные данные до: Не доступно"

        # Информация о прогнозируемом максимуме
        percent_info_max = ""
        if max_forecast_price[timeframe] is not None and current_price[timeframe] is not None:
            percent_growth = ((max_forecast_price[timeframe] - current_price[timeframe]) / current_price[timeframe]) * 100
            percent_info_max = f"\nОжидаемый процентный рост (максимум): {percent_growth:.2f}%"

        # Информация о прогнозируемом минимуме
        percent_info_min = ""
        if min_forecast_price[timeframe] is not None and current_price[timeframe] is not None:
            percent_decline = ((current_price[timeframe] - min_forecast_price[timeframe]) / current_price[timeframe]) * 100
            percent_info_min = f"\nОжидаемый процентный спад (минимум): {percent_decline:.2f}%"

        probability_request = (
            "Определи наиболее вероятный максимум и минимум цены на сутки вперед и оцени вероятность достижения этих уровней. "
            "Вероятность можно выразить в процентах (от 0% до 100%) или в терминах (очень низкая, низкая, средняя, высокая, очень высокая). "
            "Также предложи оценку вероятности роста выше текущей цены и спада ниже текущей цены."
        )

        trading_info = ""
        if current_price and timeframe in current_price and timeframe in max_forecast_price and timeframe in min_forecast_price and max_forecast_price[timeframe] is not None and min_forecast_price[timeframe] is not None:
            trading_info = (
                f"\nПредположим: "
                f"Точка входа (для покупки): {current_price[timeframe]:.4f}, "
                f"Тейк-профит: {max_forecast_price[timeframe]:.4f}, "
                f"Стоп-лосс: {current_price[timeframe] * 0.95:.4f}; "
                f"Точка входа (для продажи/шорт): {current_price[timeframe]:.4f}, "
                f"Тейк-профит: {min_forecast_price[timeframe]:.4f}, "
                f"Стоп-лосс: {current_price[timeframe] * 1.05:.4f}"
            )

        realistic_info = ""
        if realistic_percentage and timeframe in realistic_percentage:
            if realistic_percentage[timeframe]['max'] is not None:
                realistic_info_max = f"\nНаиболее реалистичное ожидаемое изменение цены за сутки (рост): {realistic_percentage[timeframe]['max']:.2f}%"
            else:
                realistic_info_max = "\nНаиболее реалистичное ожидаемое изменение цены за сутки (рост): Не определено"

            if realistic_percentage[timeframe]['min'] is not None:
                realistic_info_min = f"\nНаиболее реалистичное ожидаемое изменение цены за сутки (спад): {realistic_percentage[timeframe]['min']:.2f}%"
            else:
                realistic_info_min = "\nНаиболее реалистичное ожидаемое изменение цены за сутки (спад): Не определено"

            realistic_info = realistic_info_max + realistic_info_min

        # Включаем инструкции непосредственно в сообщение пользователя
        prompt = (
            f"Ты аналитик финансовых рынков. Проведи глубокий анализ для {symbol} в таймфрейме {timeframe}, используя Prophet, LightGBM, CatBoost, RandomForest, LSTM, SARIMA, XGBoost, ансамбль моделей, а также учти сезонность, объемы, обновление данных. "
            "Дай оценку вероятности достижения прогнозируемого максимума и минимума в процентах или в качественных терминах. "
            "Укажи ориентировочные точки входа, выхода и стоп-лосс, если они целесообразны."
            f"{price_info}\n{forecast_date_info}{percent_info_max}{percent_info_min}{trading_info}{realistic_info}\n{probability_request}"
        )

        if timeframe in data and data[timeframe] is not None and not data[timeframe].empty:
            prompt += f"\n\nТаймфрейм {timeframe}, последние 10 строк данных:\n"
            # Выводим только ключевые технические индикаторы и паттерны
            technical_cols = ['Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Doji',
                              'Morning_Star', 'Evening_Star', 'Shooting_Star',
                              'Three_White_Soldiers', 'Three_Black_Crows', 'spike',
                              'RSI', 'MACD_hist', 'ATR', 'vol_ratio', 'EMA_10', 'EMA_50']
            available_cols = [col for col in technical_cols if col in data[timeframe].columns]
            if available_cols:
                prompt += data[timeframe][available_cols].tail(10).to_string(index=False)
            else:
                prompt += "Нет доступных технических индикаторов для отображения."

        # Добавляем прогнозы моделей
        models_to_include = ['Prophet', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'SARIMA', 'XGBoost']
        for model_name in models_to_include:
            forecast_df_max = None
            forecast_df_min = None

            # Получение прогнозов для максимума
            if model_name == 'RandomForest':
                forecast_df_max = forecasts_random_forest_max.get(f'RandomForest_{timeframe}')
            elif model_name == 'XGBoost':
                forecast_df_max = forecasts_xgboost_max.get(f'XGBoost_{timeframe}')
            elif model_name == 'LightGBM':
                forecast_df_max = forecasts_lightgbm_max.get(f'LightGBM_{timeframe}')
            elif model_name == 'CatBoost':
                forecast_df_max = forecasts_catboost_max.get(f'CatBoost_{timeframe}')
            elif model_name == 'LSTM':
                forecast_df_max = forecasts_lstm_max.get(f'LSTM_{timeframe}')
            elif model_name == 'SARIMA':
                forecast_df_max = forecasts_sarima_max.get(f'SARIMA_{timeframe}')
            elif model_name == 'Prophet':
                forecast_df_max = forecasts_max.get(f'Prophet_{timeframe}')

            # Получение прогнозов для минимума
            if model_name == 'RandomForest':
                forecast_df_min = forecasts_random_forest_min.get(f'RandomForest_{timeframe}')
            elif model_name == 'XGBoost':
                forecast_df_min = forecasts_xgboost_min.get(f'XGBoost_{timeframe}')
            elif model_name == 'LightGBM':
                forecast_df_min = forecasts_lightgbm_min.get(f'LightGBM_{timeframe}')
            elif model_name == 'CatBoost':
                forecast_df_min = forecasts_catboost_min.get(f'CatBoost_{timeframe}')
            elif model_name == 'LSTM':
                forecast_df_min = forecasts_lstm_min.get(f'LSTM_{timeframe}')
            elif model_name == 'SARIMA':
                forecast_df_min = forecasts_sarima_min.get(f'SARIMA_{timeframe}')
            elif model_name == 'Prophet':
                forecast_df_min = forecasts_min.get(f'Prophet_{timeframe}')

            # Добавление информации о статусе модели
            model_status = model_statuses.get(timeframe, {}).get(model_name, "Не запущена")
            model_status_min = model_statuses.get(timeframe, {}).get(f"{model_name}_Min", "Не запущена")

            # Добавление прогноза для максимума
            if forecast_df_max is not None and not forecast_df_max.empty:
                # Извлечение прогнозов
                if model_name == 'Prophet':
                    yhat_col = f'Прогноз_Prophet_{timeframe}'
                    yhat_upper_col = f'Прогноз_Prophet_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_Prophet_Lower_{timeframe}'
                elif model_name == 'LightGBM':
                    yhat_col = f'Прогноз_LightGBM_{timeframe}'
                    yhat_upper_col = f'Прогноз_LightGBM_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_LightGBM_Lower_{timeframe}'
                elif model_name == 'CatBoost':
                    yhat_col = f'Прогноз_CatBoost_{timeframe}'
                    yhat_upper_col = f'Прогноз_CatBoost_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_CatBoost_Lower_{timeframe}'
                elif model_name == 'RandomForest':
                    yhat_col = f'Прогноз_RF_{timeframe}'
                    yhat_upper_col = f'Прогноз_RF_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_RF_Lower_{timeframe}'
                elif model_name == 'LSTM':
                    yhat_col = f'Прогноз_LSTM_{timeframe}'
                    yhat_upper_col = f'Прогноз_LSTM_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_LSTM_Lower_{timeframe}'
                elif model_name == 'SARIMA':
                    yhat_col = f'Прогноз_SARIMA_{timeframe}'
                    yhat_upper_col = f'Прогноз_SARIMA_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_SARIMA_Lower_{timeframe}'
                elif model_name == 'XGBoost':
                    yhat_col = f'Прогноз_XGBoost_{timeframe}'
                    yhat_upper_col = f'Прогноз_XGBoost_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_XGBoost_Lower_{timeframe}'
                else:
                    yhat_col = None
                    yhat_upper_col = None
                    yhat_lower_col = None

                if yhat_col and yhat_upper_col and yhat_lower_col in forecast_df_max.columns:
                    yhat = forecast_df_max[yhat_col].iloc[0]
                    yhat_upper = forecast_df_max[yhat_upper_col].iloc[0]
                    yhat_lower = forecast_df_max[yhat_lower_col].iloc[0]
                    prompt += f"\n\nПрогноз ({model_name}):\nПрогноз: {yhat:.4f}\nПрогноз Верхний: {yhat_upper:.4f}\nПрогноз Нижний: {yhat_lower:.4f}"
                    logger.info(f"Прогноз для модели {model_name} добавлен в запрос.")
                else:
                    prompt += f"\n\nПрогноз ({model_name}): Недоступен."
                    logger.warning(f"Прогноз для модели {model_name} отсутствует для таймфрейма {timeframe}.")
            else:
                prompt += f"\n\nПрогноз ({model_name}): Недоступен."
                logger.warning(f"Прогноз для модели {model_name} отсутствует для таймфрейма {timeframe}.")

            # Добавление прогноза для минимума
            if forecast_df_min is not None and not forecast_df_min.empty:
                # Извлечение прогнозов
                if model_name == 'Prophet':
                    yhat_col = f'Прогноз_Prophet_{timeframe}'
                    yhat_upper_col = f'Прогноз_Prophet_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_Prophet_Lower_{timeframe}'
                elif model_name == 'LightGBM':
                    yhat_col = f'Прогноз_LightGBM_{timeframe}'
                    yhat_upper_col = f'Прогноз_LightGBM_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_LightGBM_Lower_{timeframe}'
                elif model_name == 'CatBoost':
                    yhat_col = f'Прогноз_CatBoost_{timeframe}'
                    yhat_upper_col = f'Прогноз_CatBoost_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_CatBoost_Lower_{timeframe}'
                elif model_name == 'RandomForest':
                    yhat_col = f'Прогноз_RF_{timeframe}'
                    yhat_upper_col = f'Прогноз_RF_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_RF_Lower_{timeframe}'
                elif model_name == 'LSTM':
                    yhat_col = f'Прогноз_LSTM_{timeframe}'
                    yhat_upper_col = f'Прогноз_LSTM_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_LSTM_Lower_{timeframe}'
                elif model_name == 'SARIMA':
                    yhat_col = f'Прогноз_SARIMA_{timeframe}'
                    yhat_upper_col = f'Прогноз_SARIMA_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_SARIMA_Lower_{timeframe}'
                elif model_name == 'XGBoost':
                    yhat_col = f'Прогноз_XGBoost_{timeframe}'
                    yhat_upper_col = f'Прогноз_XGBoost_Upper_{timeframe}'
                    yhat_lower_col = f'Прогноз_XGBoost_Lower_{timeframe}'
                else:
                    yhat_col = None
                    yhat_upper_col = None
                    yhat_lower_col = None

                if yhat_col and yhat_upper_col and yhat_lower_col in forecast_df_min.columns:
                    yhat = forecast_df_min[yhat_col].iloc[0]
                    yhat_upper = forecast_df_min[yhat_upper_col].iloc[0]
                    yhat_lower = forecast_df_min[yhat_lower_col].iloc[0]
                    prompt += f"\n\nПрогноз (Минимум {model_name}):\nПрогноз: {yhat:.4f}\nПрогноз Верхний: {yhat_upper:.4f}\nПрогноз Нижний: {yhat_lower:.4f}"
                    logger.info(f"Прогноз (минимум) для модели {model_name} добавлен в запрос.")
                else:
                    prompt += f"\n\nПрогноз (Минимум {model_name}): Недоступен."
                    logger.warning(f"Прогноз (минимум) для модели {model_name} отсутствует для таймфрейма {timeframe}.")
            else:
                prompt += f"\n\nПрогноз (Минимум {model_name}): Недоступен."
                logger.warning(f"Прогноз (минимум) для модели {model_name} отсутствует для таймфрейма {timeframe}.")

        # Добавляем ансамбль прогнозов, если доступен
        # Объединение прогнозов максимумов и минимумов
        all_forecasts_max = {
            'Prophet': forecasts_max.get(f'Prophet_{timeframe}'),
            'LightGBM': forecasts_lightgbm_max.get(f'LightGBM_{timeframe}'),
            'CatBoost': forecasts_catboost_max.get(f'CatBoost_{timeframe}'),
            'RandomForest': forecasts_random_forest_max.get(f'RandomForest_{timeframe}'),
            'LSTM': forecasts_lstm_max.get(f'LSTM_{timeframe}'),
            'SARIMA': forecasts_sarima_max.get(f'SARIMA_{timeframe}'),
            'XGBoost': forecasts_xgboost_max.get(f'XGBoost_{timeframe}')
        }

        all_forecasts_min = {
            'Prophet_Min': forecasts_min.get(f'Prophet_{timeframe}'),
            'LightGBM_Min': forecasts_lightgbm_min.get(f'LightGBM_{timeframe}'),
            'CatBoost_Min': forecasts_catboost_min.get(f'CatBoost_{timeframe}'),
            'RandomForest_Min': forecasts_random_forest_min.get(f'RandomForest_{timeframe}'),
            'LSTM_Min': forecasts_lstm_min.get(f'LSTM_{timeframe}'),
            'SARIMA_Min': forecasts_sarima_min.get(f'SARIMA_{timeframe}'),
            'XGBoost_Min': forecasts_xgboost_min.get(f'XGBoost_{timeframe}')
        }

        ensemble_df_max = combine_forecasts_max(all_forecasts_max)
        ensemble_df_min = combine_forecasts_min(all_forecasts_min)

        if ensemble_df_max is not None and not ensemble_df_max.empty and ensemble_df_min is not None and not ensemble_df_min.empty:
            prompt += f"\n\nАнсамбль прогнозов (среднее всех моделей):\n{ensemble_df_max.to_string(index=False)}"
            prompt += f"\n\nАнсамбль прогнозов (минимум, среднее всех моделей):\n{ensemble_df_min.to_string(index=False)}"
            logger.info("Ансамбль прогнозов добавлен в запрос.")
        else:
            prompt += f"\n\nАнсамбль прогнозов не доступен."
            logger.warning(f"Ансамбль прогнозов не был создан для таймфрейма {timeframe}.")

        # Логируем длину промпта для отладки
        prompt_length = len(prompt.split())
        logger.info(f"Длина промпта: {prompt_length} слов")

        try:
            response = openai.ChatCompletion.create(
                model="chatgpt-4o-latest",  # Исправлено на корректное название модели
                messages=[
                    {"role": "user", "content": prompt}  # Удален 'system' роль
                ],
                temperature=0.7,
                max_tokens=3000
            )
            logger.info("Анализ с OpenAI успешно выполнен.")
            analysis = response['choices'][0]['message']['content']
            analyses.append((timeframe, analysis))
        except openai.error.RateLimitError as e:
            logger.error(f"Ошибка анализа с OpenAI: {e}")
            analyses.append((timeframe, "Ошибка анализа с OpenAI: превышен лимит запросов."))
        except openai.error.InvalidRequestError as e:
            logger.error(f"Ошибка анализа с OpenAI: {e}")
            analyses.append((timeframe, f"Ошибка анализа с OpenAI: {e}"))
        except Exception as e:
            logger.error(f"Ошибка анализа с OpenAI: {e}")
            analyses.append((timeframe, f"Ошибка анализа с OpenAI: {e}"))

    return analyses

def analyze_overall_with_openai(
    data,
    symbol,
    forecasts_max,
    forecasts_lightgbm_max,
    forecasts_catboost_max,
    forecasts_random_forest_max,
    forecasts_lstm_max=None,
    forecasts_sarima_max=None,
    forecasts_xgboost_max=None,
    forecasts_min=None,
    forecasts_lightgbm_min=None,
    forecasts_catboost_min=None,
    forecasts_random_forest_min=None,
    forecasts_lstm_min=None,
    forecasts_sarima_min=None,
    forecasts_xgboost_min=None,
    current_price=None,
    max_forecast_price=None,
    min_forecast_price=None,
    realistic_percentage=None,
    forecast_dates=None,
    model_statuses=None
):
    analyses = []

    # Вычисление среднего ансамбля всех таймфреймов
    overall_ensemble_max = []
    overall_ensemble_min = []
    for timeframe in data.keys():
        ensemble_df_max = combine_forecasts_max({
            'Prophet': forecasts_max.get(f'Prophet_{timeframe}'),
            'LightGBM': forecasts_lightgbm_max.get(f'LightGBM_{timeframe}'),
            'CatBoost': forecasts_catboost_max.get(f'CatBoost_{timeframe}'),
            'RandomForest': forecasts_random_forest_max.get(f'RandomForest_{timeframe}'),
            'LSTM': forecasts_lstm_max.get(f'LSTM_{timeframe}'),
            'SARIMA': forecasts_sarima_max.get(f'SARIMA_{timeframe}'),
            'XGBoost': forecasts_xgboost_max.get(f'XGBoost_{timeframe}')
        })
        if ensemble_df_max is not None and not ensemble_df_max.empty:
            overall_ensemble_max.append(ensemble_df_max['Прогноз_Ансамбль_Max'].iloc[-1])

        ensemble_df_min = combine_forecasts_min({
            'Prophet_Min': forecasts_min.get(f'Prophet_{timeframe}'),
            'LightGBM_Min': forecasts_lightgbm_min.get(f'LightGBM_{timeframe}'),
            'CatBoost_Min': forecasts_catboost_min.get(f'CatBoost_{timeframe}'),
            'RandomForest_Min': forecasts_random_forest_min.get(f'RandomForest_{timeframe}'),
            'LSTM_Min': forecasts_lstm_min.get(f'LSTM_{timeframe}'),
            'SARIMA_Min': forecasts_sarima_min.get(f'SARIMA_{timeframe}'),
            'XGBoost_Min': forecasts_xgboost_min.get(f'XGBoost_{timeframe}')
        })
        if ensemble_df_min is not None and not ensemble_df_min.empty:
            overall_ensemble_min.append(ensemble_df_min['Прогноз_Ансамбль_Min'].iloc[-1])

    if overall_ensemble_max and overall_ensemble_min:
        overall_forecast_max = np.mean(overall_ensemble_max)
        overall_forecast_min = np.mean(overall_ensemble_min)
        # Определение временного шага для общего прогноза (используем наиболее длинный таймфрейм)
        # В данном случае "1d"
        last_times = [df_combined['timestamp'].iloc[-1] for df_combined in data.values()]
        last_time = max(last_times)
        forecast_time_max = last_time + timedelta(days=1)  # Предполагаем дневной шаг для максимума
        forecast_time_min = last_time + timedelta(days=1)  # Предполагаем дневной шаг для минимума

        df_overall_forecast_max = pd.DataFrame({
            'Время': [forecast_time_max],
            'Прогноз_Ансамбль_Общий_Max': [overall_forecast_max]
        })

        df_overall_forecast_min = pd.DataFrame({
            'Время': [forecast_time_min],
            'Прогноз_Ансамбль_Общий_Min': [overall_forecast_min]
        })

        # Подготовка промпта для общего прогноза
        prompt_overall = (
            f"Ты аналитик финансовых рынков. Проведи глубокий анализ для {symbol} на основе прогнозов всех таймфреймов. "
            "Дай оценку вероятности достижения общего прогнозируемого максимума и минимума в процентах или в качественных терминах. "
            "Укажи ориентировочные точки входа, выхода и стоп-лосс, если они целесообразны."
        )

        prompt_overall += f"\n\nОбщий прогноз на основе всех таймфреймов (Максимум):\n{df_overall_forecast_max.to_string(index=False)}"
        prompt_overall += f"\n\nОбщий прогноз на основе всех таймфреймов (Минимум):\n{df_overall_forecast_min.to_string(index=False)}"

        # Логируем длину промпта
        prompt_length_overall = len(prompt_overall.split())
        logger.info(f"Длина промпта для общего прогноза: {prompt_length_overall} слов")

        try:
            response_overall = openai.ChatCompletion.create(
                model="gpt-4",  # Исправлено на корректное название модели
                messages=[
                    {"role": "user", "content": prompt_overall}  # Удален 'system' роль
                ],
                temperature=0.7,
                max_tokens=1500
            )
            logger.info("Анализ общего прогноза с OpenAI успешно выполнен.")
            analysis_overall = response_overall['choices'][0]['message']['content']
            analyses.append(("Общий", analysis_overall))
        except openai.error.RateLimitError as e:
            logger.error(f"Ошибка анализа общего прогноза с OpenAI: {e}")
            analyses.append(("Общий", "Ошибка анализа общего прогноза с OpenAI: превышен лимит запросов."))
        except openai.error.InvalidRequestError as e:
            logger.error(f"Ошибка анализа общего прогноза с OpenAI: {e}")
            analyses.append(("Общий", f"Ошибка анализа общего прогноза с OpenAI: {e}"))
        except Exception as e:
            logger.error(f"Ошибка анализа общего прогноза с OpenAI: {e}")
            analyses.append(("Общий", f"Ошибка анализа общего прогноза с OpenAI: {e}"))

    else:
        logger.warning("Общий прогноз не был создан из-за отсутствия достаточных данных.")

    return analyses

def save_results_to_file(file_path, results):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for symbol, analyses in results.items():
                for timeframe, analysis in analyses:
                    f.write(f"Результаты для {symbol} ({timeframe}):\n")
                    f.write(f"{analysis}\n")
                    f.write("\n" + "-" * 50 + "\n")
        logger.info(f"Результаты успешно сохранены в файл: {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов в файл: {e}")

def main():
    # Добавьте ETH/USDT в список пар
    pairs = ["ADA/USDT", "ADA/USDT"]  # Замените на нужные пары
    timeframes = ["1h", "1d"]  # Добавлен таймфрейм "1d"

    results = {}

    for symbol in pairs:
        combined_data = {}
        forecasts_max = {}
        forecasts_min = {}
        forecasts_lightgbm_max = {}
        forecasts_lightgbm_min = {}
        forecasts_catboost_max = {}
        forecasts_catboost_min = {}
        forecasts_random_forest_max = {}
        forecasts_random_forest_min = {}
        forecasts_lstm_max = {}
        forecasts_lstm_min = {}
        forecasts_sarima_max = {}
        forecasts_sarima_min = {}
        forecasts_xgboost_max = {}
        forecasts_xgboost_min = {}
        current_price = {}
        max_forecast_price = {}
        min_forecast_price = {}
        realistic_percentage = {}
        forecast_dates = {}
        model_statuses = {}

        for timeframe in timeframes:
            logger.info(f"Начало обработки {symbol} ({timeframe}).")
            df = fetch_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                df = apply_technical_indicators(df)

                # Добавление дополнительных технических индикаторов
                df['EMA_10'] = ta.EMA(df['close'], timeperiod=10)
                df['EMA_50'] = ta.EMA(df['close'], timeperiod=50)

                combined_data[timeframe] = df

                # Инициализация модели статусов для этого таймфрейма
                if timeframe not in model_statuses:
                    model_statuses[timeframe] = {}

                # Генерация прогнозов для максимума
                # Prophet
                logger.info(f"Начало прогнозирования с Prophet для {timeframe} (максимум).")
                prophet_fc_max = forecast_with_prophet(df, symbol, timeframe)
                if prophet_fc_max is not None:
                    forecasts_max[f'Prophet_{timeframe}'] = prophet_fc_max
                    model_statuses[timeframe]['Prophet'] = "Запущена успешно"
                    logger.info(f"Prophet прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['Prophet'] = "Не запущена"

                # LightGBM
                logger.info(f"Начало прогнозирования с LightGBM для {timeframe} (максимум).")
                lgb_fc_max = forecast_with_lightgbm(df, symbol, timeframe)
                if lgb_fc_max is not None:
                    forecasts_lightgbm_max[f'LightGBM_{timeframe}'] = lgb_fc_max
                    model_statuses[timeframe]['LightGBM'] = "Запущена успешно"
                    logger.info(f"LightGBM прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['LightGBM'] = "Не запущена"

                # CatBoost
                logger.info(f"Начало прогнозирования с CatBoost для {timeframe} (максимум).")
                cb_fc_max = forecast_with_catboost(df, symbol, timeframe)
                if cb_fc_max is not None:
                    forecasts_catboost_max[f'CatBoost_{timeframe}'] = cb_fc_max
                    model_statuses[timeframe]['CatBoost'] = "Запущена успешно"
                    logger.info(f"CatBoost прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['CatBoost'] = "Не запущена"

                # RandomForest
                logger.info(f"Начало прогнозирования с Random Forest для {timeframe} (максимум).")
                rf_fc_max = forecast_with_random_forest(df, symbol, timeframe)
                if rf_fc_max is not None:
                    forecasts_random_forest_max[f'RandomForest_{timeframe}'] = rf_fc_max
                    model_statuses[timeframe]['RandomForest'] = "Запущена успешно"
                    logger.info(f"RandomForest прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['RandomForest'] = "Не запущена"

                # LSTM
                logger.info(f"Начало прогнозирования с LSTM для {timeframe} (максимум).")
                lstm_fc_max = forecast_with_lstm(df, symbol, timeframe)
                if lstm_fc_max is not None:
                    forecasts_lstm_max[f'LSTM_{timeframe}'] = lstm_fc_max
                    model_statuses[timeframe]['LSTM'] = "Запущена успешно"
                    logger.info(f"LSTM прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['LSTM'] = "Не запущена"

                # SARIMA
                logger.info(f"Начало прогнозирования с SARIMA для {timeframe} (максимум).")
                sarima_fc_max = forecast_with_sarima(df, symbol, timeframe)
                if sarima_fc_max is not None:
                    forecasts_sarima_max[f'SARIMA_{timeframe}'] = sarima_fc_max
                    model_statuses[timeframe]['SARIMA'] = "Запущена успешно"
                    logger.info(f"SARIMA прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['SARIMA'] = "Не запущена"

                # XGBoost
                logger.info(f"Начало прогнозирования с XGBoost для {timeframe} (максимум).")
                xgb_fc_max = forecast_with_xgboost(df, symbol, timeframe)
                if xgb_fc_max is not None:
                    forecasts_xgboost_max[f'XGBoost_{timeframe}'] = xgb_fc_max
                    model_statuses[timeframe]['XGBoost'] = "Запущена успешно"
                    logger.info(f"XGBoost прогноз (максимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['XGBoost'] = "Не запущена"

                # Генерация прогнозов для минимума (аналогично максимуму)
                # Prophet
                logger.info(f"Начало прогнозирования с Prophet для {timeframe} (минимум).")
                prophet_fc_min = forecast_with_prophet(df, symbol, timeframe)
                if prophet_fc_min is not None:
                    forecasts_min[f'Prophet_{timeframe}'] = prophet_fc_min
                    model_statuses[timeframe]['Prophet_Min'] = "Запущена успешно"
                    logger.info(f"Prophet прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['Prophet_Min'] = "Не запущена"

                # LightGBM
                logger.info(f"Начало прогнозирования с LightGBM для {timeframe} (минимум).")
                lgb_fc_min = forecast_with_lightgbm(df, symbol, timeframe)
                if lgb_fc_min is not None:
                    forecasts_lightgbm_min[f'LightGBM_{timeframe}'] = lgb_fc_min
                    model_statuses[timeframe]['LightGBM_Min'] = "Запущена успешно"
                    logger.info(f"LightGBM прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['LightGBM_Min'] = "Не запущена"

                # CatBoost
                logger.info(f"Начало прогнозирования с CatBoost для {timeframe} (минимум).")
                cb_fc_min = forecast_with_catboost(df, symbol, timeframe)
                if cb_fc_min is not None:
                    forecasts_catboost_min[f'CatBoost_{timeframe}'] = cb_fc_min
                    model_statuses[timeframe]['CatBoost_Min'] = "Запущена успешно"
                    logger.info(f"CatBoost прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['CatBoost_Min'] = "Не запущена"

                # RandomForest
                logger.info(f"Начало прогнозирования с Random Forest для {timeframe} (минимум).")
                rf_fc_min = forecast_with_random_forest(df, symbol, timeframe)
                if rf_fc_min is not None:
                    forecasts_random_forest_min[f'RandomForest_{timeframe}'] = rf_fc_min
                    model_statuses[timeframe]['RandomForest_Min'] = "Запущена успешно"
                    logger.info(f"RandomForest прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['RandomForest_Min'] = "Не запущена"

                # LSTM
                logger.info(f"Начало прогнозирования с LSTM для {timeframe} (минимум).")
                lstm_fc_min = forecast_with_lstm(df, symbol, timeframe)
                if lstm_fc_min is not None:
                    forecasts_lstm_min[f'LSTM_{timeframe}'] = lstm_fc_min
                    model_statuses[timeframe]['LSTM_Min'] = "Запущена успешно"
                    logger.info(f"LSTM прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['LSTM_Min'] = "Не запущена"

                # SARIMA
                logger.info(f"Начало прогнозирования с SARIMA для {timeframe} (минимум).")
                sarima_fc_min = forecast_with_sarima(df, symbol, timeframe)
                if sarima_fc_min is not None:
                    forecasts_sarima_min[f'SARIMA_{timeframe}'] = sarima_fc_min
                    model_statuses[timeframe]['SARIMA_Min'] = "Запущена успешно"
                    logger.info(f"SARIMA прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['SARIMA_Min'] = "Не запущена"

                # XGBoost
                logger.info(f"Начало прогнозирования с XGBoost для {timeframe} (минимум).")
                xgb_fc_min = forecast_with_xgboost(df, symbol, timeframe)
                if xgb_fc_min is not None:
                    forecasts_xgboost_min[f'XGBoost_{timeframe}'] = xgb_fc_min
                    model_statuses[timeframe]['XGBoost_Min'] = "Запущена успешно"
                    logger.info(f"XGBoost прогноз (минимум) для {timeframe} добавлен.")
                else:
                    model_statuses[timeframe]['XGBoost_Min'] = "Не запущена"

                # Вывод технических индикаторов в консоль
                print(f"Технические индикаторы и паттерны для {symbol} ({timeframe}):")
                technical_cols = ['Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Doji',
                                  'Morning_Star', 'Evening_Star', 'Shooting_Star',
                                  'Three_White_Soldiers', 'Three_Black_Crows', 'spike',
                                  'RSI', 'MACD_hist', 'ATR', 'vol_ratio', 'EMA_10', 'EMA_50']
                available_cols = [col for col in technical_cols if col in df.columns]
                if available_cols:
                    print(df[available_cols].tail(10))
                else:
                    print("Нет доступных технических индикаторов для отображения.")
                print("\n")
            else:
                logger.warning(f"Нет данных для {symbol} ({timeframe})")

        # Анализ по каждому таймфрейму
        for timeframe in timeframes:
            if timeframe in combined_data:
                df_combined = combined_data[timeframe]
                current_price[timeframe] = df_combined['close'].iloc[-1]
                forecast_dates[timeframe] = df_combined['timestamp'].iloc[-1]

                # Получение максимальной прогнозируемой цены из всех моделей для конкретного таймфрейма
                max_forecast_price[timeframe] = None
                for model_forecast in [forecasts_max, forecasts_lightgbm_max, forecasts_catboost_max, forecasts_random_forest_max, forecasts_lstm_max, forecasts_sarima_max, forecasts_xgboost_max]:
                    if isinstance(model_forecast, dict) and len(model_forecast) > 0:
                        key = list(model_forecast.keys())[0]
                        forecast_df = model_forecast.get(key, None)
                        if forecast_df is not None and not forecast_df.empty:
                            yhat_col = None
                            # Определяем колонку прогноза
                            if 'Прогноз_Prophet_' in key:
                                yhat_col = f'Прогноз_Prophet_{timeframe}'
                            elif 'Прогноз_LightGBM_' in key:
                                yhat_col = f'Прогноз_LightGBM_{timeframe}'
                            elif 'Прогноз_CatBoost_' in key:
                                yhat_col = f'Прогноз_CatBoost_{timeframe}'
                            elif 'Прогноз_RF_' in key:
                                yhat_col = f'Прогноз_RF_{timeframe}'
                            elif 'Прогноз_LSTM_' in key:
                                yhat_col = f'Прогноз_LSTM_{timeframe}'
                            elif 'Прогноз_SARIMA_' in key:
                                yhat_col = f'Прогноз_SARIMA_{timeframe}'
                            elif 'Прогноз_XGBoost_' in key:
                                yhat_col = f'Прогноз_XGBoost_{timeframe}'
                            else:
                                continue

                            model_max = forecast_df[yhat_col].max()
                            if model_max and (max_forecast_price[timeframe] is None or model_max > max_forecast_price[timeframe]):
                                max_forecast_price[timeframe] = model_max

                # Получение минимальной прогнозируемой цены из всех моделей для конкретного таймфрейма
                min_forecast_price[timeframe] = None
                for model_forecast in [forecasts_min, forecasts_lightgbm_min, forecasts_catboost_min, forecasts_random_forest_min, forecasts_lstm_min, forecasts_sarima_min, forecasts_xgboost_min]:
                    if isinstance(model_forecast, dict) and len(model_forecast) > 0:
                        key = list(model_forecast.keys())[0]
                        forecast_df = model_forecast.get(key, None)
                        if forecast_df is not None and not forecast_df.empty:
                            yhat_col = None
                            # Определяем колонку прогноза
                            if 'Прогноз_Prophet_' in key:
                                yhat_col = f'Прогноз_Prophet_{timeframe}'
                            elif 'Прогноз_LightGBM_' in key:
                                yhat_col = f'Прогноз_LightGBM_{timeframe}'
                            elif 'Прогноз_CatBoost_' in key:
                                yhat_col = f'Прогноз_CatBoost_{timeframe}'
                            elif 'Прогноз_RF_' in key:
                                yhat_col = f'Прогноз_RF_{timeframe}'
                            elif 'Прогноз_LSTM_' in key:
                                yhat_col = f'Прогноз_LSTM_{timeframe}'
                            elif 'Прогноз_SARIMA_' in key:
                                yhat_col = f'Прогноз_SARIMA_{timeframe}'
                            elif 'Прогноз_XGBoost_' in key:
                                yhat_col = f'Прогноз_XGBoost_{timeframe}'
                            else:
                                continue

                            model_min = forecast_df[yhat_col].min()
                            if model_min and (min_forecast_price[timeframe] is None or model_min < min_forecast_price[timeframe]):
                                min_forecast_price[timeframe] = model_min

                if max_forecast_price[timeframe] and min_forecast_price[timeframe]:
                    max_growth = ((max_forecast_price[timeframe] - current_price[timeframe]) / current_price[timeframe]) * 100
                    min_decline = ((current_price[timeframe] - min_forecast_price[timeframe]) / current_price[timeframe]) * 100
                    stop_loss_entry = current_price[timeframe] * 0.95
                    take_profit_entry = max_forecast_price[timeframe] * 0.98
                    stop_loss_exit_time = current_price[timeframe] * 1.05
                    take_profit_exit_time = min_forecast_price[timeframe] * 0.98

                    # Определение наиболее реалистичного прогноза (среднее ансамбля)
                    ensemble_df_max = combine_forecasts_max({
                        'Prophet': forecasts_max.get(f'Prophet_{timeframe}'),
                        'LightGBM': forecasts_lightgbm_max.get(f'LightGBM_{timeframe}'),
                        'CatBoost': forecasts_catboost_max.get(f'CatBoost_{timeframe}'),
                        'RandomForest': forecasts_random_forest_max.get(f'RandomForest_{timeframe}'),
                        'LSTM': forecasts_lstm_max.get(f'LSTM_{timeframe}'),
                        'SARIMA': forecasts_sarima_max.get(f'SARIMA_{timeframe}'),
                        'XGBoost': forecasts_xgboost_max.get(f'XGBoost_{timeframe}')
                    })
                    if ensemble_df_max is not None and not ensemble_df_max.empty:
                        realistic_forecast_max = ensemble_df_max['Прогноз_Ансамбль_Max'].iloc[-1]
                        realistic_percentage_max = ((realistic_forecast_max - current_price[timeframe]) / current_price[timeframe]) * 100
                    else:
                        realistic_percentage_max = None

                    ensemble_df_min = combine_forecasts_min({
                        'Prophet_Min': forecasts_min.get(f'Prophet_{timeframe}'),
                        'LightGBM_Min': forecasts_lightgbm_min.get(f'LightGBM_{timeframe}'),
                        'CatBoost_Min': forecasts_catboost_min.get(f'CatBoost_{timeframe}'),
                        'RandomForest_Min': forecasts_random_forest_min.get(f'RandomForest_{timeframe}'),
                        'LSTM_Min': forecasts_lstm_min.get(f'LSTM_{timeframe}'),
                        'SARIMA_Min': forecasts_sarima_min.get(f'SARIMA_{timeframe}'),
                        'XGBoost_Min': forecasts_xgboost_min.get(f'XGBoost_{timeframe}')
                    })
                    if ensemble_df_min is not None and not ensemble_df_min.empty:
                        realistic_forecast_min = ensemble_df_min['Прогноз_Ансамбль_Min'].iloc[-1]
                        realistic_percentage_min = ((current_price[timeframe] - realistic_forecast_min) / current_price[timeframe]) * 100
                    else:
                        realistic_percentage_min = None

                    realistic_percentage[timeframe] = {
                        'max': realistic_percentage_max,
                        'min': realistic_percentage_min
                    }

                    print(f"Пара: {symbol}")
                    print(f"Таймфрейм: {timeframe}")
                    print(f"Текущая цена: {current_price[timeframe]:.4f}")
                    print(f"Прогнозируемая максимальная цена: {max_forecast_price[timeframe]:.4f}")
                    print(f"Прогнозируемая минимальная цена: {min_forecast_price[timeframe]:.4f}")
                    print(f"Ожидаемый рост (максимум): {max_growth:.2f}%")
                    print(f"Ожидаемый спад (минимум): {min_decline:.2f}%")
                    print(f"Точка входа (покупка): {current_price[timeframe]:.4f}")
                    print(f"Тейк-профит (покупка): {take_profit_entry:.4f}")
                    print(f"Стоп-лосс (покупка): {stop_loss_entry:.4f}")
                    print(f"Точка входа (продажа/шорт): {current_price[timeframe]:.4f}")
                    print(f"Тейк-профит (продажа/шорт): {take_profit_exit_time:.4f}")
                    print(f"Стоп-лосс (продажа/шорт): {stop_loss_exit_time:.4f}")
                    if realistic_percentage_max is not None and realistic_percentage_min is not None:
                        print(f"Наиболее реалистичное ожидаемое изменение цены за сутки (рост): {realistic_percentage_max:.2f}%")
                        print(f"Наиболее реалистичное ожидаемое изменение цены за сутки (спад): {realistic_percentage_min:.2f}%\n")
                    else:
                        print("Наиболее реалистичное ожидаемое изменение цены за сутки: Не определено\n")
                else:
                    logger.warning(f"Не удалось определить прогнозируемые максимумы и минимумы для {symbol} ({timeframe}).")
            else:
                logger.warning(f"Таймфрейм {timeframe} не найден в combined_data для {symbol}.")

        # Подготовка данных для анализа с OpenAI
        logger.info(f"Начало анализа с OpenAI для {symbol}.")
        analysis = analyze_with_openai(
            data=combined_data,
            symbol=symbol,
            forecasts_max=forecasts_max,
            forecasts_lightgbm_max=forecasts_lightgbm_max,
            forecasts_catboost_max=forecasts_catboost_max,
            forecasts_random_forest_max=forecasts_random_forest_max,
            forecasts_lstm_max=forecasts_lstm_max,
            forecasts_sarima_max=forecasts_sarima_max,
            forecasts_xgboost_max=forecasts_xgboost_max,
            forecasts_min=forecasts_min,
            forecasts_lightgbm_min=forecasts_lightgbm_min,
            forecasts_catboost_min=forecasts_catboost_min,
            forecasts_random_forest_min=forecasts_random_forest_min,
            forecasts_lstm_min=forecasts_lstm_min,
            forecasts_sarima_min=forecasts_sarima_min,
            forecasts_xgboost_min=forecasts_xgboost_min,
            current_price=current_price,
            max_forecast_price=max_forecast_price,
            min_forecast_price=min_forecast_price,
            realistic_percentage=realistic_percentage,
            forecast_dates=forecast_dates,
            model_statuses=model_statuses
        )
        if analysis:
            results[symbol] = analysis

    # Анализ общего прогноза
    logger.info(f"Начало общего анализа с OpenAI для {pairs}.")
    overall_analysis = analyze_overall_with_openai(
        data=combined_data,
        symbol=symbol,
        forecasts_max=forecasts_max,
        forecasts_lightgbm_max=forecasts_lightgbm_max,
        forecasts_catboost_max=forecasts_catboost_max,
        forecasts_random_forest_max=forecasts_random_forest_max,
        forecasts_lstm_max=forecasts_lstm_max,
        forecasts_sarima_max=forecasts_sarima_max,
        forecasts_xgboost_max=forecasts_xgboost_max,
        forecasts_min=forecasts_min,
        forecasts_lightgbm_min=forecasts_lightgbm_min,
        forecasts_catboost_min=forecasts_catboost_min,
        forecasts_random_forest_min=forecasts_random_forest_min,
        forecasts_lstm_min=forecasts_lstm_min,
        forecasts_sarima_min=forecasts_sarima_min,
        forecasts_xgboost_min=forecasts_xgboost_min,
        current_price=current_price,
        max_forecast_price=max_forecast_price,
        min_forecast_price=min_forecast_price,
        realistic_percentage=realistic_percentage,
        forecast_dates=forecast_dates,
        model_statuses=model_statuses
    )
    if overall_analysis:
        if symbol in results:
            results[symbol].extend(overall_analysis)
        else:
            results[symbol] = overall_analysis

    file_path = r"D:\\7\\results.txt"
    save_results_to_file(file_path, results)

if __name__ == "__main__":
    # Проверка версий библиотек
    import xgboost
    import sklearn

    logger.info(f"Версия XGBoost: {xgboost.__version__}")
    logger.info(f"Версия scikit-learn: {sklearn.__version__}")

    main()
