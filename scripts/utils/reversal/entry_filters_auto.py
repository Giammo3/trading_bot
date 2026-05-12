def filter_rsi_14(row): return row.get('rsi_14', 0) > 47.386613
def filter_acceleration_5_norm(row): return row.get('acceleration_5_norm', 0) < -0.054741
def filter_wick_body_ratio(row): return row.get('wick_body_ratio', 0) > 3.394454

entry_filters_reversal = [
    filter_rsi_14,
    filter_acceleration_5_norm,
    filter_wick_body_ratio,
]
