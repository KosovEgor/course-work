import os
import time
import pickle
import warnings
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse

warnings.filterwarnings("ignore")

from model import fetch_sber_tinvest, FEATURES
from PDT import predict

PDT_MODEL_PATH = "./output/sber_pdt_model.pkl"
CB_MODEL_PATH  = "./output/sber_catboost_model.pkl"

app = FastAPI()

_pdt_tree = None
_cb_model = None

_CACHE: dict = {}
_CACHE_TTL = 300

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.monotonic() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    return None

def _cache_set(key: str, val):
    _CACHE[key] = {"val": val, "ts": time.monotonic()}


def get_pdt_tree():
    global _pdt_tree
    if _pdt_tree is None:
        with open(PDT_MODEL_PATH, "rb") as f:
            _pdt_tree = pickle.load(f)
    return _pdt_tree


def get_cb_model():
    global _cb_model
    if _cb_model is None:
        with open(CB_MODEL_PATH, "rb") as f:
            _cb_model = pickle.load(f)
    return _cb_model


def compute_features_live(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sh = (df['High'] > df['High'].shift(1)) & \
         (df['High'] > df['High'].shift(2)) & \
         (df['High'] > df['High'].shift(3))
    df['Swing_High'] = df['High'].where(sh).ffill()
    sl = (df['Low'] < df['Low'].shift(1)) & \
         (df['Low'] < df['Low'].shift(2)) & \
         (df['Low'] < df['Low'].shift(3))
    df['Swing_Low'] = df['Low'].where(sl).ffill()
    df['Dist_To_Swing_High'] = df['Close'] - df['Swing_High']
    df['Dist_To_Swing_Low']  = df['Close'] - df['Swing_Low']
    r_hi = df['High'].rolling(5).max().shift(1)
    r_lo = df['Low'].rolling(5).min().shift(1)
    df['Order_Block'] = ((r_hi - r_lo) < (df['Close'] * 0.002).shift(1)).astype(int)
    df['MA_20'] = df['Close'].rolling(20).mean().shift(1)
    df['MA_50'] = df['Close'].rolling(50).mean().shift(1)
    df['diff']  = df['MA_20'] - df['MA_50']
    delta = df['Close'].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = (100 - 100 / (1 + avg_gain / avg_loss)).shift(1)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def df_to_chart_data(df: pd.DataFrame) -> list:

    ts = (pd.to_datetime(df["time"]).astype("int64") // 1_000_000).tolist()
    opens = df["Open"].round(2).tolist()
    highs = df["High"].round(2).tolist()
    lows = df["Low"].round(2).tolist()
    closes = df["Close"].round(2).tolist()
    return [
        {"x": t, "y": [o, h, l, c]}
        for t, o, h, l, c in zip(ts, opens, highs, lows, closes)
    ]


def build_prediction_text(row: pd.Series, signal: str, model_name: str) -> str:
    if row['RSI'] > 70:
        rsi_desc = "рынок перекуплен"
    elif row['RSI'] < 30:
        rsi_desc = "рынок перепродан"
    else:
        rsi_desc = "нейтральное состояние рынка"
    return (
        f"Рекомендация {model_name}: {signal}\n\n"
        f"Ключевая статистика\n"
        f"Цена: {row['Close']:.2f} ₽\n"
        f"RSI: {row['RSI']:.1f} - {rsi_desc}\n"
        f"MA20: {row['MA_20']:.2f} - средняя цена за последние 20 свечей\n"
        f"MA50: {row['MA_50']:.2f} - средняя цена за последние 50 свечей\n"
    )


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str):
    df = _cache_get("raw_df")
    if df is None:
        df = fetch_sber_tinvest(days=0.5)
        _cache_set("raw_df", df)

    chart_data = _cache_get("chart_data")
    if chart_data is None:
        chart_data = df_to_chart_data(df)
        _cache_set("chart_data", chart_data)

    df_feat = _cache_get("df_feat")
    if df_feat is None:
        df_feat = compute_features_live(df)
        _cache_set("df_feat", df_feat)

    latest = df_feat.iloc[-1]
    x = [float(latest[f]) for f in FEATURES]

    if ticker == "SBER_CB":
        model = get_cb_model()
        pred = int(model.predict([x])[0])
        model_name = "CatBoost"
    else:
        tree = get_pdt_tree()
        pred = int(predict(tree, x))
        model_name = "PDT"

    signal = "КУПИТЬ" if pred == 1 else "ПРОДАТЬ"
    prediction = build_prediction_text(latest, signal, model_name)

    return JSONResponse({"data": chart_data, "prediction": prediction})


@app.get("/")
async def root():
    return FileResponse("index.html")