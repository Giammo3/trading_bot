import csv #serve per leggere/scrivere file csv
import os #serve per fare controlli sul file

def salva_su_csv(file_path, riga_dati):
    file_esiste= os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_esiste:
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        writer.writerow([
            riga_dati['timestamp'],
            riga_dati['open'],
            riga_dati['high'],
            riga_dati['low'],
            riga_dati['close'],
            riga_dati['volume']
        ])