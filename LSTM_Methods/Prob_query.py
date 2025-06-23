import pandas as pd
import numpy as np
from PinballLSTM_Interval_Engine import make_feature_matrix, run_window, build_quants

CFG = dict(
    horizon     = 7,          # 预测步长 H
    train_fr    = 0.6,
    cali_fr     = 0.2,
    seq_len     = 60,           #用多少数据预测
    epochs      = 30,
    batch       = 256,
    hidden      = 32,
    dropout     = 0.2,
    conf_lvl    = 0.92,
    n_sim       = 40_000,
    use_cols = ['close','Volume USDT','tradecount','IV','Fear_index'], # ← None = 默认抓取 raw 中所有数值列；也可传列表手选

    # === 分位区间可调 ===
    quant_lo    = 0.10,        # ⬅︎ 最小分位 (含)
    quant_hi    = 0.90,        # ⬅︎ 最大分位 (含)
    quant_step  = 0.01,        # 步长
)
CFG['quants'] = build_quants(CFG)

conf_lvl= CFG['conf_lvl']*100
horizon     = CFG['horizon']

raw = (
    pd.read_csv(r'D:\prediction_market_data\crypto_data\merged_btc_data.csv', parse_dates=["date"])
      .set_index("date")
      .sort_index()
      .loc[lambda df: ~df.index.duplicated()]  # 去重
)

'''
data must have: columns={date,close}
'''
df_feat = make_feature_matrix(raw, CFG)
feat_cols = [c for c in df_feat.columns if c != 'y']
res = run_window(df_feat, feat_cols, CFG)

sims_log = res['sims'][-1]  # log-price Monte-Carlo
sims = np.exp(sims_log)  # ← 转回价格空间

low, high = 55_000, 65_000
prob = ((sims >= low) & (sims <= high)).mean()
print(f"{horizon}天后落在 [{low},{high}] 的概率: {prob:.2%}")
L90 = np.exp(res['L'][-1]);
U90 = np.exp(res['U'][-1])
print(f"{conf_lvl}% 预测区间: [{L90:.0f}, {U90:.0f}]")

'''
Sliding_windows :  SampleX:seq_len, Outputy:t+horizon
'''
