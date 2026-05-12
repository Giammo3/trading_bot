from twelvedata import TDClient
from configv1 import API_KEY, CURRENCY_PAIR, INTERVAL

def get_forex_data():
    td = TDClient(apikey=API_KEY)

    base, quote = CURRENCY_PAIR.split('/')

    try:

        data = td.time_series(
            symbol=f'{base}/{quote}',
            interval=INTERVAL,
            outputsize=1
        ).as_json()[0]

        return {
            'timestamp': data['datetime'],
            'open': float(data['open']),
            'high': float(data['high']),
            'low': float(data['low']),
            'close': float(data['close']),
            'volume': 0.0
        }
        
    except Exception as e:
        print('[!] Errore durante la richiesta:', e)
        return None




