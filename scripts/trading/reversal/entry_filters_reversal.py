
def filter_volatility(row):
    return row.get('volatility_10', 0) > 0.0003

def filter_acceleration(row):
    return row.get('acceleration_5_norm', 0) < 0

def filter_adx(row):
    return row.get('adx', 0) > 15

def filter_zscore(row):
    return row.get('zscore_ma50_filtered', 0) > -1

def filter_wick_ratio(row):
    return row.get('wick_body_ratio', 0) > 4.5

entry_filters_reversal = [
    filter_volatility,
    filter_acceleration,
    filter_adx,
    filter_zscore,
    filter_wick_ratio
]