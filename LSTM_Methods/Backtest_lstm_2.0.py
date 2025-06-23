"""
BTC Interval-Engine ★ Backtest Edition v3 – Δ-log-price
-------------------------------------------------------
目标 = 未来 H 日 log-return；网络输出收益分位点；推断时平移回价格。
"""
import random, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from tqdm import trange

warnings.filterwarnings("ignore")
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ————————— feature —————————
PRICE_WINDOW = 60
def default_cfg():
    return dict(
        train_fr=0.8, seq_len=60, epochs=300, batch=64,
        hidden=256, dropout=0.2, conf_lvl=0.90,
        n_sim=100000, quant_lo=0.05, quant_hi=0.95, quant_step=0.005,
        train_window_days=720, horizon=3,        early_stop_patience=10,   # val loss 连续10个epoch无改善就停止
        lr_reduce_patience=5,     # val loss 连续5个epoch无改善就降lr
        lr_reduce_factor=0.5,     # 每次 lr *= 0.5
    )
# ————————— utilities —————————
class PinballLSTM(nn.Module):
    def __init__(self, n_feat, n_q, cfg):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, cfg['hidden'], 2,
                            batch_first=True, dropout=cfg['dropout'])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg['hidden']),
            nn.Linear(cfg['hidden'], cfg['hidden']//2), nn.ReLU(),
            nn.Linear(cfg['hidden']//2, n_q)
        )
    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])

def build_quants(cfg):
    return np.round(np.arange(cfg['quant_lo'], cfg['quant_hi'] + 1e-9, cfg['quant_step']), 5)

def qloss(pred, targ, quants):
    q = torch.as_tensor(quants, device=pred.device)
    e = targ.unsqueeze(1) - pred
    return torch.mean(torch.maximum((q - 1)*e, q*e))

def make_feature_matrix(df):
    out = df.copy()
    if 'iv' in out and out['iv'].max() > 2:
        out['iv'] /= 100
    if 'vol_spike_10' in out:
        out['vol_spike_10'] = np.log1p(out['vol_spike_10'])
    return out

# ————————— interval generation —————————
def generate_intervals(p, step=2000):
    base_left = (int(p)//step)*step
    base_right = base_left + step
    bounds = [base_left - 3*step, base_left - 2*step, base_left - step,
              base_left, base_right, base_right + step]
    return [
        (-np.inf, bounds[0]), (bounds[0], bounds[1]),
        (bounds[1], bounds[2]), (bounds[2], bounds[3]),
        (bounds[3], bounds[4]), (bounds[4], bounds[5]),
        (bounds[5], np.inf)
    ]

# ————————— model training —————————
def train_model(X, Y, tr_mask, X_val, Y_val, cfg, device):
    quants = build_quants(cfg)
    mdl = PinballLSTM(X.shape[2], len(quants), cfg).to(device)
    opt = optim.AdamW(mdl.parameters(), lr=3e-4)
    # Reduce-On-Plateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=cfg['lr_reduce_factor'],
        patience=cfg['lr_reduce_patience'],
        verbose=False
    )
    best_val = float('inf')
    no_improve = 0

    for epoch in range(cfg['epochs']):
        mdl.train()
        idx = np.random.permutation(np.where(tr_mask)[0])
        for i in range(0, len(idx), cfg['batch']):
            b = idx[i:i+cfg['batch']]
            xb = torch.tensor(X[b], dtype=torch.float32, device=device)
            yb = torch.tensor(Y[b], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = qloss(mdl(xb), yb, quants)
            loss.backward()
            opt.step()

        # —— 验证集上计算 Pinball Loss ——
        mdl.eval()
        with torch.no_grad():
            xb_val = torch.tensor(X_val, dtype=torch.float32, device=device)
            yb_val = torch.tensor(Y_val, dtype=torch.float32, device=device)
            val_loss = qloss(mdl(xb_val), yb_val, quants).item()

        # 调度 & 早停逻辑
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg['early_stop_patience']:
                print(f"EarlyStopping at epoch {epoch}, val_loss={val_loss:.6f}")
                break

    return mdl


# ————————— CQR 校准 —————————
def calibrate_cqr(mdl, X_cal, Y_cal, quants, conf_lvl, device):
    mdl.eval()
    with torch.no_grad():
        preds = mdl(torch.tensor(X_cal, dtype=torch.float32, device=device)).cpu().numpy()
    preds = np.maximum.accumulate(preds, axis=1)
    err_lo = np.maximum(Y_cal - preds[:,0], 0)
    err_hi = np.maximum(preds[:,-1] - Y_cal, 0)
    k = int(np.ceil((len(err_lo)+1)*(1-(1-conf_lvl)/2))) - 1
    return np.sort(err_lo)[k], np.sort(err_hi)[k]

# ————————— interval inference —————————
def infer_intervals(mdl, win_pred_z, δ_lo, δ_hi, quants, cfg, device, last_price):
    mdl.eval()
    with torch.no_grad():
        q_r = mdl(torch.tensor(win_pred_z[None], dtype=torch.float32, device=device)).cpu().numpy()[0]
    q_r = np.maximum.accumulate(q_r)
    q_r[0] -= δ_lo; q_r[-1] += δ_hi
    q_r = np.maximum.accumulate(q_r)
    ln_pt = np.log(last_price)
    q_lp = ln_pt + q_r
    u = np.random.rand(cfg['n_sim'])
    idx = np.clip(np.searchsorted(quants, u)-1, 0, len(quants)-2)
    sims_lp = q_lp[idx] + (u-quants[idx])*(q_lp[idx+1]-q_lp[idx])/cfg['quant_step']
    sims_price = np.exp(sims_lp)
    intervals = generate_intervals(last_price)
    probs = [((sims_price >= lo) & (sims_price <= hi)).mean()
             for lo, hi in intervals]
    return intervals, probs

# ————————— hit evaluation —————————
def evaluate_hits(intervals, probs, true_price):
    order = np.argsort(probs)[::-1]
    best, second = order[0], order[1]
    lo_b, hi_b = intervals[best]; lo_s, hi_s = intervals[second]
    hit_b = lo_b <= true_price <= hi_b
    hit_s = lo_s <= true_price <= hi_s
    true_bin = next(i for i,(lo,hi) in enumerate(intervals) if lo<= true_price <= hi)
    return hit_b, hit_s, probs[true_bin]

# ————————— backtest —————————
def backtest(csv_path, cfg, verbose=True,
             rolling_cols=None, static_cols=None, raw_cols=None):
    rolling_cols = rolling_cols or []
    static_cols  = static_cols or []
    raw_cols     = raw_cols or []

    df = (pd.read_csv(csv_path, parse_dates=['date'])
            .rename(columns=str.strip).rename(columns=str.lower)
            .set_index('date').sort_index())
    df = make_feature_matrix(df)

    L, H = cfg['seq_len'], cfg['horizon']
    look = cfg['train_window_days'] + L + PRICE_WINDOW + H
    start, end = look-1, len(df)-1-H

    hit_b_tot = hit_s_tot = total = 0
    prob20_cnt = 0
    prob_sum = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quants = build_quants(cfg)

    for i in trange(start, end+1, desc='Rolling', ncols=70):
        seg = df.iloc[i-look+1:i+1]
        # ② Rolling Z-score
        base = seg.iloc[:-H]
        mu = base[rolling_cols].rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW).mean().reindex(seg.index, method='ffill')
        sd = base[rolling_cols].rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW).std().add(1e-12).reindex(seg.index, method='ffill')
        for c in rolling_cols:
            seg[f'{c}_z'] = (seg[c] - mu[c]) / sd[c]
        feat = seg[[f'{c}_z' for c in rolling_cols] + static_cols + raw_cols].dropna()

        # ④ 标签
        price = seg['log_price']
        df_y = feat.iloc[:-L].copy()
        df_y = feat.iloc[:-H].copy()
        df_y['y'] = (price.shift(-H) - price).reindex(df_y.index)

        split = int(len(df_y) * cfg['train_fr'])
        # — 静态标准化（新增）—
        μ_s = df_y[static_cols].iloc[:split].mean()
        σ_s = df_y[static_cols].iloc[:split].std().add(1e-12)
        df_y[static_cols] = (df_y[static_cols] - μ_s) / σ_s
        feat[static_cols] = (feat[static_cols] - μ_s) / σ_s

        # ⑥ 滑窗训练集 (fixed: only feature columns)
        feature_cols = [c for c in df_y.columns if c != 'y']
        X, Y = [], []
        for j in range(L, len(df_y)):
            X.append(df_y[feature_cols].iloc[j - L:j].values)
            Y.append(df_y['y'].iloc[j])

        X = np.array(X)
        Y = np.array(Y)

        tr_mask = np.arange(len(X)) < (split - L)


        # ⑦ 模型生命周期
        X_val, Y_val = X[~tr_mask], Y[~tr_mask]
        mdl = train_model(X, Y, tr_mask, X_val, Y_val, cfg, device)
        X_cal, Y_cal = X[~tr_mask], Y[~tr_mask]
        δ_lo,δ_hi = calibrate_cqr(mdl, X_cal, Y_cal, quants, cfg['conf_lvl'], device)
        win_z     = feat.tail(L).values
        last_p    = np.exp(seg['log_price'].iloc[-1])
        ints, ps  = infer_intervals(mdl, win_z, δ_lo, δ_hi, quants, cfg, device, last_p)
        true_p = np.exp(df['log_price'].iloc[i + cfg['horizon']])
        hb, hs, pc= evaluate_hits(ints, ps, true_p)

        hit_b_tot += hb; hit_s_tot += hs; total += 1; prob_sum += pc
        if pc >= 0.15 and not (hb or hs):
            prob20_cnt += 1

        if verbose:
            # 找到模型最看好的两个区间
            order = np.argsort(ps)[::-1]
            best_idx, second_idx = order[0], order[1]
            lo_b, hi_b = ints[best_idx]
            lo_s, hi_s = ints[second_idx]

            # 打印格式与第一个版本一致
            print(
                f"{seg.index[-1].date()}  "
                f"best[{lo_b:,.0f},{hi_b if hi_b != np.inf else '∞'}] "
                f"P={ps[best_idx]:.2%}  "
                f"2nd[{lo_s:,.0f},{hi_s if hi_s != np.inf else '∞'}] "
                f"P={ps[second_idx]:.2%}  "
                f"prob_correct={pc:.2%}  "
                f"hit_b={hb} hit_s={hs}  "
                f"true_price={true_p:,.0f}"
            )

    print("\n──────── SUMMARY ────────")
    print(f"Days: {total}, Hit_best: {hit_b_tot}({hit_b_tot/total:.2%}), "
          f"Hit_second: {hit_s_tot}({hit_s_tot/total:.2%}), "
          f"Avg_prob_correct: {prob_sum/total:.2%}),"
          f"Prob≥15% but not in First/Second Interval: {prob20_cnt}({prob20_cnt / total:.2%})")

# ————————— run —————————
if __name__ == '__main__':
    cfg = default_cfg()
    path = r"D:\\prediction_market_data\\crypto_data\\merged_btc3.csv"
    rolling_cols = ['log_price','hv5','iv']
    static_cols  = ['oc_ret','ema5_10','vol_spike_10','fear_index']
    raw_cols     = ['ret1_z','ret5_z','rsi14_z']
    backtest(path, cfg, verbose=True, rolling_cols=rolling_cols,
             static_cols=static_cols, raw_cols=raw_cols)

'''
──────── SUMMARY ────────
2-layerslstm 5-pre-length:Days: 645, Hit_best: 158(24.50%), Hit_second: 161(24.96%), Avg_prob_correct: 21.63%),Prob≥20% but not in First/Second Interval: 42(6.51%)

'''
