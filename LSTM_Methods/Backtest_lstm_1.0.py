"""
BTC Interval-Engine ★ Backtest Edition v3 – Δ-log-price
-------------------------------------------------------
目标 = 未来 H 日 log-return；网络输出收益分位点；推断时平移回价格。
"""
import random, warnings, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────── utilities ─────────────
def build_quants(cfg):
    return np.round(np.arange(cfg["quant_lo"],
                              cfg["quant_hi"] + 1e-9,
                              cfg["quant_step"]), 5)

class PinballLSTM(nn.Module):
    def __init__(self, n_feat, n_q, cfg):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, cfg["hidden"], 2,
                            batch_first=True, dropout=cfg["dropout"])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg["hidden"]),
            nn.Linear(cfg["hidden"], cfg["hidden"] // 2), nn.ReLU(),
            nn.Linear(cfg["hidden"] // 2, n_q)
        )
    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])

def qloss(pred, targ, quants):
    q = torch.as_tensor(quants, device=pred.device)
    e = targ.unsqueeze(1) - pred
    return torch.mean(torch.maximum((q - 1) * e, q * e))

# ───────────── feature ─────────────
PRICE_WINDOW = 90
def make_feature_matrix(raw):
    out = pd.DataFrame(index=raw.index)
    out["close"] = raw["close"]
    for ma in ["ma30_raw", "ma90_raw"]:
        if ma in raw:
            out[f"dist_{ma[:-4]}"] = (raw["close"] - raw[ma]) / raw[ma]
    out["log_close"] = np.log(raw["close"].clip(1e-8))
    out["log_vol"]   = np.log(raw["volume_usdt"] + 1)
    for col in ["hv20_raw", "fear_idx", "iv", "rsi14_raw"]:
        if col in raw: out[col] = raw[col]
    if "iv" in out and out["iv"].max() > 2:  out["iv"] /= 100
    if "rsi14_raw" in out:                   out["rsi14_raw"] /= 100
    return out

import numpy as np

import numpy as np


def generate_intervals(p: float, step: int = 2_000):
    """
    构造 7 个对称区间；除两端无穷尾外，所有有限区间长度均为 `step`
    (默认 2 000)。中央单一区间覆盖当前价 p。

    例：p = 81 350, step = 2 000
      → (-∞,78k], (78k,80k], (80k,82k], (82k,84k], (84k,86k],
         (86k,88k], (88k,∞)
    """
    # 1) 最近的 step 整数下界 / 上界
    base_left = (int(p) // step) * step  # ≤ p
    base_right = base_left + step  # > p

    bounds = [  # 6 条有限边界
        base_left - 3 * step,  # -3s
        base_left - 2 * step,  # -2s
        base_left - step,  # -1s
        base_left,  # 0   （正好 = 中央区间左端）
        base_right,  # +1s
        base_right + step  # +2s
    ]

    # 7 个区间：左尾 + 3 × step + 中央 + 2 × step + 右尾
    intervals = [
        (-np.inf, bounds[0]),
        (bounds[0], bounds[1]),
        (bounds[1], bounds[2]),
        (bounds[2], bounds[3]),  # 宽度 = step
        (bounds[3], bounds[4]),  # ← 中央区间（含 p）
        (bounds[4], bounds[5]),
        (bounds[5], np.inf)
    ]
    return intervals


# ───────────── config ─────────────
def default_cfg():
    return dict(train_fr=0.7, seq_len=60, epochs=150, batch=64 ,
                hidden=256, dropout=0.2, conf_lvl=0.90,
                n_sim=100_000, quant_lo=0.05, quant_hi=0.95, quant_step=0.005,
                train_window_days=720, horizon=5)

# ───────────── backtest ─────────────
def backtest(csv_path, cfg, verbose=True):
    raw = (pd.read_csv(csv_path, parse_dates=["date"])
             .rename(columns=str.strip).rename(columns=str.lower)
             .set_index("date").sort_index())

    L, H = cfg["seq_len"], cfg["horizon"]
    look = cfg["train_window_days"] + L + PRICE_WINDOW + H
    start, end = look-1, len(raw)-1-H

    hit_best = hit_second = total = 0
    prob_correct_sum = 0.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    quants = build_quants(cfg)

    for i in trange(start, end+1, desc="Rolling", ncols=70):

        seg = raw.iloc[i-look+1:i+1]
        feat = make_feature_matrix(seg).dropna()
        base_feat = feat.iloc[:-H] if H else feat          # 无未来信息

        roll_cols = ["log_close", "hv20_raw", "iv"]
        roll_mu = (base_feat[roll_cols]
                   .rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW)
                   .mean().reindex(feat.index, method="ffill"))
        roll_sd = (base_feat[roll_cols]
                   .rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW)
                   .std().add(1e-12).reindex(feat.index, method="ffill"))
        for col in roll_cols:
            if col in feat:
                feat[f"{col}_z"] = (feat[col]-roll_mu[col]) / roll_sd[col]

        keep = ["log_close_z", "hv20_raw_z", "iv_z", "log_vol",
                "dist_ma30", "dist_ma90", "rsi14_raw", "fear_idx"]
        feat = feat[[c for c in keep if c in feat]].dropna()

        # ── 标签：Δ log-price ────────────────────────────────────────────
        ln_close_seg = np.log(seg["close"])
        df_y = feat.iloc[:-L].copy()                         # 对齐 X_window 终点
        df_y["y"] = (ln_close_seg.shift(-H) - ln_close_seg).reindex(df_y.index)
        df_y.dropna(subset=["y"], inplace=True)
        split = int(len(df_y) * cfg["train_fr"])

        # ── 静态列 z-score ───────────────────────────────────────────────
        vol_cols = ["log_vol"]
        other_cols = [c for c in feat.columns
                      if c not in vol_cols + ["log_close_z","hv20_raw_z","iv_z"]]
        μv, σv = df_y[vol_cols].iloc[:split].mean(), df_y[vol_cols].iloc[:split].std()
        μo, σo = df_y[other_cols].iloc[:split].mean(), df_y[other_cols].iloc[:split].std()

        def scale(df):
            d = df.copy()
            if vol_cols:  d[vol_cols]  = (d[vol_cols]-μv)/(σv+1e-12)
            if other_cols:d[other_cols]=(d[other_cols]-μo)/(σo+1e-12)
            return d

        z_all      = scale(df_y.drop(columns="y"))
        # 建议使用（含当日）：
        win_pred_z = scale(feat.tail(L))

        #win_pred_z = scale(feat.iloc[-(L+1):-1])             # 最后 L 行，不含当日

        # ── 训练集：滑窗 ────────────────────────────────────────────────
        X, Y = [], []
        for j in range(L, len(z_all)-H):
            X.append(z_all.iloc[j-L:j].values)
            Y.append(df_y["y"].iloc[j])
        X, Y = np.asarray(X), np.asarray(Y)
        tr_mask = np.arange(len(X)) < split-L

        mdl = PinballLSTM(z_all.shape[1], len(quants), cfg).to(dev)
        opt = optim.AdamW(mdl.parameters(), lr=3e-4)
        for _ in range(cfg["epochs"]):
            mdl.train()
            perm = np.random.permutation(np.where(tr_mask)[0])
            for k in range(0, len(perm), cfg["batch"]):
                b = perm[k:k+cfg["batch"]]
                xb = torch.tensor(X[b], dtype=torch.float32, device=dev)
                yb = torch.tensor(Y[b], dtype=torch.float32, device=dev)
                opt.zero_grad(); qloss(mdl(xb), yb, quants).backward(); opt.step()

        # ── CQR 校准（仍在收益空间）──────────────────────────────────────
        mdl.eval()
        with torch.no_grad():
            ca_pred = mdl(torch.tensor(X[~tr_mask], dtype=torch.float32, device=dev)).cpu().numpy()
        ca_pred = np.maximum.accumulate(ca_pred, 1)
        err_lo  = np.maximum(Y[~tr_mask] - ca_pred[:,0], 0)
        err_hi  = np.maximum(ca_pred[:,-1] - Y[~tr_mask], 0)
        k = int(np.ceil((len(err_lo)+1)*(1-(1-cfg["conf_lvl"])/2))) - 1
        δ_lo, δ_hi = np.sort(err_lo)[k], np.sort(err_hi)[k]

        # ── 预测 ΔlogP, 调整, 单调化 ───────────────────────────────────
        with torch.no_grad():
            q_r = np.maximum.accumulate(
                mdl(torch.tensor(win_pred_z.values[None], dtype=torch.float32, device=dev))
                .cpu().numpy()[0])
        q_r[0] -= δ_lo;  q_r[-1] += δ_hi
        q_r = np.maximum.accumulate(q_r)

        # ── 转回价格空间并蒙特卡洛 ────────────────────────────────────
        p_t   = seg["close"].iloc[-1]
        ln_pt = np.log(p_t)
        q_lp  = ln_pt + q_r

        u   = torch.rand(cfg["n_sim"]).numpy()
        idx = np.searchsorted(quants, u, side="right") - 1
        idx = idx.clip(0, len(quants)-2)
        sims_lp = q_lp[idx] + (u-quants[idx]) * (q_lp[idx+1]-q_lp[idx]) / cfg["quant_step"]
        sims_price = np.exp(sims_lp)

        # ── 统计区间命中 ───────────────────────────────────────────────
        intervals = generate_intervals(p_t)
        probs = [((sims_price>=lo)&(sims_price<=hi)).mean() for lo,hi in intervals]
        best_idx, second_idx = np.argsort(probs)[::-1][:2]

        lo_best, hi_best   = intervals[best_idx]
        lo_sec,  hi_sec    = intervals[second_idx]

        true_price = raw["close"].iloc[i+H]
        hit_b = lo_best <= true_price <= hi_best
        hit_s = lo_sec  <= true_price <= hi_sec
        hit_best += hit_b; hit_second += hit_s; total += 1

        true_bin = next(j for j,(lo,hi) in enumerate(intervals) if lo<=true_price<=hi)
        prob_correct_sum += probs[true_bin]

        if verbose:
            tqdm.write(f"{seg.index[-1].date()}  "
                       f"best[{lo_best:,.0f},{hi_best if hi_best!=np.inf else '∞'}] "
                       f"P={probs[best_idx]:.2%}  "
                       f"2nd[{lo_sec:,.0f},{hi_sec if hi_sec!=np.inf else '∞'}] "
                       f"P={probs[second_idx]:.2%}  "
                       f"prob_correct={probs[true_bin]:.2%}  "
                       f"hit_b={hit_b} hit_s={hit_s}  "
                       f"true_price={true_price:,.0f}")
            del mdl
            torch.cuda.empty_cache()

    print("\n──────── SUMMARY ────────")
    print(f"Days                 : {total}")
    print(f"Hits best-bin        : {hit_best}  ({hit_best/total:.2%})")
    print(f"Hits second-bin      : {hit_second} ({hit_second/total:.2%})")
    print(f"Avg prob_correct_bin : {prob_correct_sum/total:.2%}")

# ───────────── run ─────────────
if __name__ == "__main__":
    cfg = default_cfg()
    csv_path = r"D:\prediction_market_data\crypto_data\merged_btc_feat.csv"
    backtest(csv_path, cfg, verbose=True)
'''
result:──────── SUMMARY ────────
Days                 : 602
Hits best-bin        : 147  (24.42%)
Hits second-bin      : 106 (17.61%)
Avg prob_correct_bin : 21.23%
'''