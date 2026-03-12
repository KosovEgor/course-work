from t_tech.invest import Client, CandleInterval
from t_tech.invest import *
from t_tech.invest.utils import *
from t_tech.invest.schemas import *
 
from datetime import *
import mplfinance as mpf
import pandas as pd


token = 't.s0_spTuGC3vjsUuRS61TQrZiyw9cx1gc_UcQdqSjnTbi4CND4e4eCgoM71KuzG71VzU2doMtXyYcY8WQYQCl5A'
figi_sber = 'BBG004730N88'
candles = []


def main():
    with Client(token) as client:
        res = client.get_all_candles(
            instrument_id=figi_sber,
            interval=CandleInterval.CANDLE_INTERVAL_4_HOUR,
            from_= now() - timedelta(days=10)
        )
        for c in res:
            candles.append({
            'time':c.time,
            'open':c.open.units,
            'close':c.close.units,
            'low':c.low.units,
            'high':c.high.units,
            'volume':c.volume
            })
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    print(1)
    mpf.plot(df, type='candle', volume=True, style='charles', savefig='sber_candles.png')

if __name__ == '__main__':
    main()