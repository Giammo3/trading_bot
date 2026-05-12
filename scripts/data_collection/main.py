
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import time
from dati import get_forex_data
from salvatore import salva_su_csv

#diamo il nome al nostro file csv 
file_path = 'forex_data.csv'


while True:
        dati = get_forex_data()

        if dati:
               salva_su_csv(file_path, dati)
               print(f"[✓] Salvato alle {dati['timestamp']} | Close = {dati['close']}")

        else:
                print('[!] Dati non disponibili, riprovo...')
                
                
        time.sleep(300)





