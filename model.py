import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from t_tech.invest import Client, CandleInterval
from t_tech.invest.utils import now

from PDT import predict, build_pdt

from catboost import CatBoostClassifier


warnings.filterwarnings('ignore')
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_sber_tinvest(days=90):

    TOKEN = 't.QuBYwxAYWh1-bOjEZdNgGtE9A5-9ojLpagprZOY2zIF_vMFtD2aqVpMDYacEPJv7O_lbHILWIMvbUNDJ5PXlgQ'
    FIGI  = 'BBG004730N88'
    candles = []
    with Client(TOKEN) as client:
        for c in client.get_all_candles(
            instrument_id=FIGI,
            interval=CandleInterval.CANDLE_INTERVAL_5_MIN,
            from_=now() - timedelta(days=days),
        ):
            candles.append({
                'time': c.time,
                'Open': float(c.open.units  + c.open.nano  / 1e9),
                'Close': float(c.close.units + c.close.nano / 1e9),
                'Low': float(c.low.units   + c.low.nano   / 1e9),
                'High': float(c.high.units  + c.high.nano  / 1e9),
                'Volume': c.volume,
            })
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def identify_swings(df, window=3):
    sh = (df['High'] > df['High'].shift(1)) & \
         (df['High'] > df['High'].shift(2)) & \
         (df['High'] > df['High'].shift(3))
    df['Swing_High'] = df['High'].where(sh).ffill()
    sl = (df['Low'] < df['Low'].shift(1)) & \
         (df['Low'] < df['Low'].shift(2)) & \
         (df['Low'] < df['Low'].shift(3))
    df['Swing_Low'] = df['Low'].where(sl).ffill()
    return df


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    RS = avg_gain / avg_loss
    return 100 - (100 / (1 + RS))


def prepare_df(df):
    df = df.copy()
    df = identify_swings(df, window=3)
    df['Dist_To_Swing_High'] = df['Close'] - df['Swing_High']
    df['Dist_To_Swing_Low']  = df['Close'] - df['Swing_Low']
    w = 5
    rng = df['Close'] * 0.002
    r_hi = df['High'].rolling(window=w).max().shift(1)
    r_lo = df['Low'].rolling(window=w).min().shift(1)
    df['Order_Block'] = ((r_hi - r_lo) < rng.shift(1)).astype(int)
    df['MA_20'] = df['Close'].rolling(window=20).mean().shift(1)
    df['MA_50'] = df['Close'].rolling(window=50).mean().shift(1)
    df['diff'] = df['MA_20'] - df['MA_50']
    df['RSI'] = compute_RSI(df['Close'], period=14).shift(1)
    df.dropna(inplace=True)
    future_horizon = 50
    df['Future_Close'] = df['Close'].shift(-future_horizon)
    df.dropna(inplace=True)
    df['Target'] = np.where(df['Future_Close'] > df['Close'], 1, 0)
    return df.reset_index(drop=True)


FEATURES = ['Dist_To_Swing_High', 'Dist_To_Swing_Low',
            'Order_Block', 'MA_20', 'MA_50', 'RSI', 'diff']


def simulate(stock_symbol, results_df, initial_balance=10_000,
             trailing_stop_percent=0.005):
    balance = initial_balance
    positions = 0
    trades = 0
    successful_trades = 0
    trade_profits = []
    portfolio_values  = []
    entry_price = None
    trailing_stop = None
    for i in range(len(results_df)):
        prediction = results_df.loc[i, 'Predicted']
        actual_price = results_df.loc[i, 'Close']
        if prediction == 1 and positions == 0:
            positions = int(balance // actual_price)
            if positions > 0:
                entry_price = actual_price
                balance -= positions * actual_price
                trailing_stop = entry_price * (1 - trailing_stop_percent)
        if positions > 0:
            new_ts = actual_price * (1 - trailing_stop_percent)
            if new_ts > trailing_stop:
                trailing_stop = new_ts
            if actual_price <= trailing_stop:
                exit_price = trailing_stop
                net_profit = positions * (exit_price - entry_price)
                balance += positions * exit_price
                pct = (net_profit / (entry_price * positions)) * 100
                trade_profits.append(pct)
                trades += 1
                if net_profit > 0:
                    successful_trades += 1
                positions = 0
        if prediction == 0 and positions > 0:
            net_profit = positions * (actual_price - entry_price)
            balance += positions * actual_price
            pct = (net_profit / (entry_price * positions)) * 100
            trade_profits.append(pct)
            trades += 1
            if net_profit > 0:
                successful_trades += 1
            positions = 0
        holdings = positions * actual_price if positions > 0 else 0
        portfolio_values.append(balance + holdings)
    if positions > 0:
        last_price = results_df.loc[len(results_df) - 1, 'Close']
        net_profit = positions * (last_price - entry_price)
        balance += positions * last_price
        pct = (net_profit / (entry_price * positions)) * 100
        trade_profits.append(pct)
        trades += 1
        if net_profit > 0:
            successful_trades += 1
        portfolio_values[-1] = balance
    final_value = balance
    total_return = ((balance - initial_balance) / initial_balance) * 100
    init_price = results_df['Close'].iloc[0]
    last_price = results_df['Close'].iloc[-1]
    bh_return = ((last_price - init_price) / init_price) * 100
    accuracy = (successful_trades / trades * 100) if trades else 0
    avg_trade_profit  = np.mean(trade_profits) if trades else 0
    pv = pd.Series(portfolio_values)
    running_max = pv.cummax()
    drawdown_pct = ((pv - running_max) / running_max) * 100
    max_drawdown = drawdown_pct.min()
    summary = {
        'Stock': stock_symbol,
        'Initial Balance': initial_balance,
        'Final Portfolio Value': round(final_value, 2),
        'Growth (%)': round(total_return, 4),
        'Max Drawdown (%)': round(max_drawdown, 4),
        'Buy and Hold Return (%)': round(bh_return, 4),
        'Trading Accuracy (%)': round(accuracy, 4),
        'Total Trades': trades,
        'Successful Trades': successful_trades,
        'Average Trade Profit (%)': round(avg_trade_profit, 4),
    }
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    return portfolio_values, drawdown_pct.tolist(), summary


def plot_portfolio(results_df, portfolio_values, symbol, out_dir):
    dates = results_df['Datetime'].values
    n = len(dates)
    init_close = results_df['Close'].iloc[0]
    bh_values  = [results_df['Close'].iloc[i] / init_close * 10_000 for i in range(n)]
    pv_pct = [(v - 10_000) / 10_000 * 100 for v in portfolio_values]
    bh_pct = [(v - 10_000) / 10_000 * 100 for v in bh_values]
    days = np.arange(1, n + 1)
    pv_s = pd.Series(pv_pct)
    bh_s = pd.Series(bh_pct)
    out_m = pv_s > bh_s
    under_m = ~out_m
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(days, pv_s, label="PDT Bot (%)", lw=2)
    ax.plot(days, bh_s, label="Buy & Hold (%)", lw=2, ls='--')
    ax.fill_between(days, bh_s, pv_s, where=out_m, alpha=0.3,
                    interpolate=True, label='Outperformance')
    ax.fill_between(days, bh_s, pv_s, where=under_m, alpha=0.3,
                    interpolate=True, label='Underperformance')
    max_ret = pv_s.max()
    max_day = days[pv_s.idxmax()]
    ax.plot(max_day, max_ret, 'o', ms=10, mfc='none', mec='black', mew=2)
    ax.annotate(f"Peak: {max_ret:.2f}%", xy=(max_day, max_ret),
                xytext=(max_day + max(1, n // 20), max_ret - 0.5),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.7))
    ax.set_xlabel('Trading Period (candles)', fontsize=14)
    ax.set_ylabel('Return vs Initial (%)', fontsize=14)
    ax.set_title(f'PDT Bot vs Buy & Hold — {symbol}', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    path = os.path.join(out_dir, 'portfolio_vs_buyhold.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_drawdown(results_df, drawdown_pct, symbol, out_dir):
    n = len(drawdown_pct)
    days = np.arange(1, n + 1)
    close = results_df['Close'].values
    cummax = np.maximum.accumulate(close)
    bh_dd = (close - cummax) / cummax * 100
    pv_dd = pd.Series(drawdown_pct)
    bh_dd = pd.Series(bh_dd)
    out_m = pv_dd > bh_dd
    under_m = ~out_m
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(days, pv_dd, label="PDT Bot Drawdown (%)", lw=2)
    ax.plot(days, bh_dd, label="Buy & Hold Drawdown (%)", lw=2, ls='--')
    ax.fill_between(days, bh_dd, pv_dd, where=out_m, alpha=0.3,
                    interpolate=True, label='Bot Less Drawdown')
    ax.fill_between(days, bh_dd, pv_dd, where=under_m, alpha=0.3,
                    interpolate=True, label='Bot More Drawdown')
    min_dd  = pv_dd.min()
    min_day = days[pv_dd.idxmin()]
    ax.plot(min_day, min_dd, 'o', ms=10, mfc='none', mec='black', mew=2)
    ax.annotate(f"Max DD: {min_dd:.2f}%", xy=(min_day, min_dd),
                xytext=(min_day + max(1, n // 20), min_dd + 0.5),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.7))
    ax.set_xlabel('Trading Period (candles)', fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=14)
    ax.set_title(f'Drawdown Analysis — {symbol}', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    path = os.path.join(out_dir, 'drawdown_chart.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    SYMBOL = 'SBER'
    SPLIT_RATIO = 0.80
    MAX_DEPTH = 20

    df_raw = fetch_sber_tinvest(60)
    df_raw.to_csv(os.path.join(OUTPUT_DIR, 'sber_data_raw.csv'), index=False)
    df = prepare_df(df_raw)
    df.to_csv(os.path.join(OUTPUT_DIR, 'sber_data_processed.csv'), index=False)

    X = df[FEATURES].values
    y = df['Target'].values

    split_idx = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pdt = build_pdt(X_train, y_train, max_depth=MAX_DEPTH)
    y_pred_pdt = [predict(pdt, x) for x in X_test]

    test_dates = df['time'].iloc[split_idx:].values
    test_close = df['Close'].iloc[split_idx:].values
    results_df_pdt = pd.DataFrame({
        'Datetime': test_dates,
        'Actual': y_test,
        'Predicted': y_pred_pdt,
        'Close': test_close,
    }).reset_index(drop=True)

    
    results_df_pdt.to_csv(os.path.join(OUTPUT_DIR, 'sber_results_test.csv'), index=False)
    print("Running trading simulation of PDT...\n")
    portfolio_values_pdt, drawdown_pct_pdt, summary_pdt = simulate(
        SYMBOL, results_df_pdt,
        initial_balance=10_000,
        trailing_stop_percent=0.005,
    )

    summary_pdt['Train Candles'] = len(X_train)
    summary_pdt['Test Candles']  = len(X_test)
    summary_pdt['Features'] = ', '.join(FEATURES)
    metrics_df = pd.DataFrame([summary_pdt])
    metrics_path = os.path.join(OUTPUT_DIR, 'sber_metrics_summary.csv')
    metrics_df.to_csv(metrics_path, index=False)
    portfolio_df = pd.DataFrame({
        'Datetime': results_df_pdt['Datetime'],
        'Portfolio_Value': portfolio_values_pdt,
    })
    portfolio_df.to_csv(os.path.join(OUTPUT_DIR, 'sber_portfolio_curve.csv'), index=False)
    plot_portfolio(results_df_pdt, portfolio_values_pdt, SYMBOL, OUTPUT_DIR)
    plot_drawdown(results_df_pdt,  drawdown_pct_pdt,     SYMBOL, OUTPUT_DIR)
    pkl_path = os.path.join(OUTPUT_DIR, 'sber_pdt_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pdt, f)
    print("\n── Key Metrics of PDT ──────────────────────────────────")
    print(f"  Growth (%):           {summary_pdt['Growth (%)']:>10.4f}%")
    print(f"  Max Drawdown (%):     {summary_pdt['Max Drawdown (%)']:>10.4f}%")
    print(f"  Buy & Hold Return:    {summary_pdt['Buy and Hold Return (%)']:>10.4f}%")
    print(f"  Trading Accuracy:     {summary_pdt['Trading Accuracy (%)']:>10.4f}%")
    print(f"  Total Trades:         {summary_pdt['Total Trades']:>10}")
    print(f"  Avg Trade Profit:     {summary_pdt['Average Trade Profit (%)']:>10.4f}%")
    print("────────────────────────────────────────────────────────\n")


    cat_model = CatBoostClassifier(
        depth=5,
        verbose=False, 
        iterations=300,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=42,
        early_stopping_rounds=30,
        subsample=0.8,
        colsample_bylevel=0.8,
        bootstrap_type='Bernoulli',
        loss_function='Logloss',
        eval_metric='AUC',
        class_weights=[1, 2]
    )

    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict(X_test)

    results_cat = pd.DataFrame({
        'Datetime': test_dates,
        'Actual': y_test,
        'Predicted': y_pred_cat,
        'Close': test_close,
    }).reset_index(drop=True)

    print("Running trading simulation of Catboost...\n")
    portfolio_values_cat, drawdown_pct_cat, summary_cat = simulate(
        SYMBOL, results_cat,
        initial_balance=10_000,
        trailing_stop_percent=0.005,
    )

    print("\n── Key Metrics of Catboost ──────────────────────────────────")
    print(f"  Growth (%):           {summary_cat['Growth (%)']:>10.4f}%")
    print(f"  Max Drawdown (%):     {summary_cat['Max Drawdown (%)']:>10.4f}%")
    print(f"  Buy & Hold Return:    {summary_cat['Buy and Hold Return (%)']:>10.4f}%")
    print(f"  Trading Accuracy:     {summary_cat['Trading Accuracy (%)']:>10.4f}%")
    print(f"  Total Trades:         {summary_cat['Total Trades']:>10}")
    print(f"  Avg Trade Profit:     {summary_cat['Average Trade Profit (%)']:>10.4f}%")
    print("────────────────────────────────────────────────────────")


    return summary_pdt


if __name__ == '__main__':
    main()