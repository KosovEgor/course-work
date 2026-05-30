import os
import pickle
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse


warnings.filterwarnings("ignore")

from model import fetch_sber_tinvest, identify_swings, compute_RSI, FEATURES
from PDT import predict

MODEL_PATH = "./output/sber_pdt_model.pkl"

app = FastAPI()

_tree = None

def get_tree():
    global _tree
    if _tree is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _tree = pickle.load(f)
    return _tree


def compute_features_live(df):
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


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str):
    df = fetch_sber_tinvest(days=90)

    chart_data = []
    for _, row in df.iterrows():
        ts_ms = int(pd.Timestamp(row["time"]).timestamp() * 1000)
        chart_data.append({
            "x": ts_ms,
            "y": [
                round(float(row["Open"]),  2),
                round(float(row["High"]),  2),
                round(float(row["Low"]),   2),
                round(float(row["Close"]), 2),
            ],
        })

    tree = get_tree()

    df_feat = compute_features_live(df)
    row = df_feat.iloc[-1]
    x = [float(row[f]) for f in FEATURES]
    pred = int(predict(tree, x))
    signal = "КУПИТЬ" if pred == 1 else "ПРОДАТЬ"
    prediction = (
        f"Рекомендация PDT : {signal}\n\n"
        f"Ключевая статистика\n"
        f"Цена: {row['Close']:.2f} ₽\n"
        f"RSI: {row['RSI']:.1f} - {'рынок перекуплен' if row['RSI'] > 70 else 'рынок перепродан' if row['RSI'] < 30 else 'нейтральное состояние рынка'}\n"
        f"MA20: {row['MA_20']:.2f} - средняя цена за последние 20 свечей\n"
        f"MA50: {row['MA_50']:.2f} - средняя цена за последние 50 свечей\n"
    )

    return JSONResponse({"data": chart_data, "prediction": prediction})


@app.get("/")
async def root():
    return FileResponse("index.html")
