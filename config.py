
# API CONFIG
API_KEY = '543d192f51cd43d8a3b1c465dff3032f'
SYMBOL = 'EUR/USD'
INTERVAL = '5min'

#DATA PATHS
RAW_DATA_PATH = 'datasets/forex/forex_features.csv'
OPTIMIZED_DATA_PATH = 'datasets/forex/forex_features_optimized.csv'

# Binary_02 model PATHS
X_TEST_BINARY_02 = 'datasets/binary_02/X_test_filtered.csv'
X_TEST_PREDICTED_BINARY = "datasets/binary_02/X_test_predicted.csv"
Y_TEST_BINARY_02 = 'datasets/binary_02/y_test.csv'
MODEL_BINARY_V1 = 'models/binary_02/model_v1.pkl'
MODEL_BINARY_V2 = 'models/binary_02/model_v2.pkl'

# Reversal model paths
X_TEST_REVERSAL = 'datasets/reversal/X_test_filtered_with_flat.csv'
Y_TEST_REVERSAL = 'datasets/reversal/y_test.csv'
MODEL_REVERSAL_V1 = 'models/reversal/model_reversal.pkl'
MODEL_REVERSAL_V2 = 'models/reversal/model_reversal_v2.pkl'
TOP_FEATURES_REVERSAL = 'datasets/reversal/top_features.csv'

# Output trading paths
X_TEST_TRADE_BINARY = 'datasets/binary_02/X_test_traded.csv'
X_TEST_TRADE_REVERSAL = 'datasets/reversal/X_test_traded.csv'

### TRADING SETTINGS ###
START_BALANCE = 10000  # saldo iniziale dei bot
STAKE = 100            # stake per singolo trade
TAKE_PROFIT_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.002    # 0.2%
LOOK_AHEAD_STEPS = 30   # numero massimo di candele da osservare dopo l'ingresso
