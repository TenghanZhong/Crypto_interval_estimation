
"""
BTC-Interval-Engine V3  (single-shot, Δ-log-price)
--------------------------------------------------
* 目标 = 未来 H 日 **log-return** ⇒ 网络输出收益分位点
* Conformal δ 在收益空间计算
* 推断时用 lnP_t 平移回未来价格分位点，再 exp 得价位
* 最后 Monte-Carlo & 二元期权式区间概率
"""

import re, random, warnings
from datetime import datetime, timedelta
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
warnings.filterwarnings("ignore")

SEED = 2025
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)

# ───────────────────────────────── Pinball-LSTM ──────────────────────────
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
    return torch.mean(torch.maximum((q - 1)*e, q*e))

# ──────────────────────────────── Feature 工具 ───────────────────────────
PRICE_WINDOW = 180
def make_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["close"] = raw["close"]
    for ma in ["ma30_raw", "ma90_raw"]:
        if ma in raw:
            out[f"dist_{ma[:-4]}"] = (raw["close"] - raw[ma]) / raw[ma]
    out["log_close"] = np.log(raw["close"].clip(1e-8))
    out["log_vol"]   = np.log(raw["volume_usdt"] + 1)
    for c in ["hv20_raw", "fear_idx", "iv", "rsi14_raw"]:
        if c in raw: out[c] = raw[c]
    if "iv" in out and out["iv"].max() > 2:  out["iv"] /= 100
    if "rsi14_raw" in out:                   out["rsi14_raw"] /= 100
    return out

def generate_intervals(p):
    e = [p-6000, p-4000, p-2000, p+2000, p+4000, p+6000]
    return [(-np.inf, e[0]), (e[0], e[1]), (e[1], e[2]),
            (e[2], e[3]), (e[3], e[4]), (e[4], e[5]), (e[5], np.inf)]

# ─────────────────────────────── Question parser ─────────────────────────
_dollar = r"\$?([0-9]+(?:,[0-9]{3})*)"
_to_f   = lambda s: float(s.replace(",", ""))

def parse_question(q):
    txt = q.lower()
    m = re.search(r"on ([a-z]+)\s+(\d{1,2})", txt)
    month, day = m.group(1).title(), int(m.group(2))
    nums = [_to_f(n) for n in re.findall(_dollar, txt)]
    if "less than"  in txt: return {"interval": (-np.inf, nums[0]), "month": month, "day": day}
    if "greater"    in txt: return {"interval": (nums[0], np.inf),  "month": month, "day": day}
    if "between"    in txt: return {"interval": (nums[0], nums[1]), "month": month, "day": day}
    raise ValueError("un-parsable question")

# ─────────────────────────────────  Main engine  ─────────────────────────
def answer_questions(questions, ts_str, csv_path, cfg, DEBUG=False):
    meta  = [parse_question(q) for q in questions]
    month, day = meta[0]["month"], meta[0]["day"]

    ts_dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
    tgt   = datetime(ts_dt.year, datetime.strptime(month, "%B").month, day)
    if tgt.date() <= ts_dt.date():
        tgt = tgt.replace(year=tgt.year + 1)
    H = (tgt.date() - ts_dt.date()).days                # 前向 horizon

    cfg = cfg.copy();  cfg.update(horizon=H)
    quants = build_quants(cfg)

    # ─── 1. 载入数据 & 特征 ─────────────────────────────────────────────
    raw = (pd.read_csv(csv_path, parse_dates=["date"])
             .rename(columns=str.strip).rename(columns=str.lower)
             .set_index("date").sort_index())
    lookback = cfg["train_window_days"] + cfg["seq_len"] + H + PRICE_WINDOW
    raw = raw.loc[ts_dt - timedelta(days=lookback): ts_dt]

    feat = make_feature_matrix(raw).dropna()

    # ─── 2. rolling-z（无未来泄漏）──────────────────────────────────────
    base = feat.iloc[:-H] if H else feat
    roll_cols = ["log_close", "hv20_raw", "iv"]
    roll_mu = (base[roll_cols]
               .rolling(PRICE_WINDOW, PRICE_WINDOW).mean()
               .reindex(feat.index, method="ffill"))
    roll_sd = (base[roll_cols]
               .rolling(PRICE_WINDOW, PRICE_WINDOW).std().add(1e-12)
               .reindex(feat.index, method="ffill"))
    for col in roll_cols:
        if col in feat:
            feat[f"{col}_z"] = (feat[col]-roll_mu[col]) / roll_sd[col]

    keep = ["log_close_z", "hv20_raw_z", "iv_z", "log_vol",
            "dist_ma30", "dist_ma90", "rsi14_raw", "fear_idx"]
    feat = feat[[c for c in keep if c in feat]].dropna()

    # ─── 3.  构造 Δ-log-return 标签 ★★ changed ─────────────────────────
    ln_close = np.log(raw["close"])
    df_y = feat.iloc[:-cfg["seq_len"]].copy()
    df_y["y"] = (ln_close.shift(-H) - ln_close).reindex(df_y.index)
    df_y.dropna(subset=["y"], inplace=True)
    split = int(len(df_y)*cfg["train_fr"])

    # ─── 4.  静态列 z-score  ───────────────────────────────────────────
    vol_cols   = ["log_vol"]
    other_cols = [c for c in feat.columns
                  if c not in vol_cols + ["log_close_z","hv20_raw_z","iv_z"]]
    μv, σv = df_y[vol_cols].iloc[:split].mean(), df_y[vol_cols].iloc[:split].std()
    μo, σo = df_y[other_cols].iloc[:split].mean(), df_y[other_cols].iloc[:split].std()

    def scale(df):
        d = df.copy()
        if vol_cols:   d[vol_cols]   = (d[vol_cols]-μv)/(σv+1e-12)
        if other_cols:d[other_cols] = (d[other_cols]-μo)/(σo+1e-12)
        return d

    z_all      = scale(df_y.drop(columns="y"))
    win_pred_z = scale(feat.tail(cfg["seq_len"]))           # ★★ changed (含当日)

    # ─── 5.  滑窗数据集 ────────────────────────────────────────────────
    L = cfg["seq_len"]
    X, Y = [], []
    for j in range(L, len(z_all)-H):                        # ★★ ensure j+H valid
        X.append( z_all.iloc[j-L:j].values )
        Y.append( df_y["y"].iloc[j] )
    X, Y = np.asarray(X), np.asarray(Y)
    tr_mask = np.arange(len(X)) < split-L

    # ─── 6.  训练 Pinball-LSTM（一次性）────────────────────────────────
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = PinballLSTM(z_all.shape[1], len(quants), cfg).to(dev)
    opt = optim.AdamW(mdl.parameters(), lr=3e-4)

    for _ in range(cfg["epochs"]):
        mdl.train()
        perm = np.random.permutation(np.where(tr_mask)[0])
        for k in range(0, len(perm), cfg["batch"]):
            b  = perm[k:k+cfg["batch"]]
            xb = torch.tensor(X[b], dtype=torch.float32, device=dev)
            yb = torch.tensor(Y[b], dtype=torch.float32, device=dev)
            opt.zero_grad();  qloss(mdl(xb), yb, quants).backward();  opt.step()

    # ─── 7.  CQR-split（收益空间）──────────────────────────────────────
    mdl.eval()
    with torch.no_grad():
        cal_pred = mdl(torch.tensor(X[~tr_mask], dtype=torch.float32, device=dev)).cpu().numpy()
    cal_pred = np.maximum.accumulate(cal_pred, 1)
    err_lo = np.maximum(Y[~tr_mask] - cal_pred[:,0], 0)
    err_hi = np.maximum(cal_pred[:,-1] - Y[~tr_mask], 0)
    k = int(np.ceil((len(err_lo)+1)*(1-(1-cfg["conf_lvl"])/2))) - 1
    δ_lo, δ_hi = np.sort(err_lo)[k], np.sort(err_hi)[k]

    # ─── 8.  预测 Δ-logP 分位点 & 调整 ────────────────────────────────
    with torch.no_grad():
        q_r = np.maximum.accumulate(
              mdl(torch.tensor(win_pred_z.values[None], dtype=torch.float32, device=dev))
              .cpu().numpy()[0])
    q_r[0] -= δ_lo;  q_r[-1] += δ_hi
    q_r = np.maximum.accumulate(q_r)

    # ─── 9.  平移回价格分位点 → Monte-Carlo ───────────────────────────
    lnP_t = np.log(raw["close"].iloc[-1])
    q_lp  = lnP_t + q_r

    u   = torch.rand(cfg["n_sim"]).numpy()
    idx = np.searchsorted(quants, u, "right") - 1
    idx = idx.clip(0, len(quants)-2)
    sims_lp = q_lp[idx] + (u-quants[idx]) * (q_lp[idx+1]-q_lp[idx]) / cfg["quant_step"]
    sims_price = np.exp(sims_lp)

    # ─── 10.  输出问题答案 ───────────────────────────────────────────
    out = {}
    for qtxt, meta_i in zip(questions, meta):
        lo, hi = meta_i["interval"]
        out[qtxt] = ((sims_price >= lo) & (sims_price <= hi)).mean()

    # 额外返回 90% 区间（可选）
    out["pred_interval"] = ( np.exp(q_lp[0]), np.exp(q_lp[-1]) )

    if DEBUG:
        print("ℹ 90% interval:", out["pred_interval"])
        print("ℹ MC median:", np.median(sims_price),
              "p5:", np.percentile(sims_price,5),
              "p95:", np.percentile(sims_price,95))
    return out

# ──────────────────────────────── Config  ────────────────────────────────
def default_cfg():
    return dict(train_fr=0.7, seq_len=60, epochs=300, batch=64,
                hidden=256, dropout=0.2,
                conf_lvl=0.90,
                n_sim=100_000, quant_lo=0.05, quant_hi=0.95, quant_step=0.005,
                train_window_days=720)

# ───────────────────────────── Example run ───────────────────────────────
if __name__ == "__main__":
    QUESTIONS = [
        "Will the price of Bitcoin be less than $93000 on May 9?",
        "Will the price of Bitcoin be between $93000 and $95000 on May 9?",
        "Will the price of Bitcoin be between $95000 and $97000 on May 9?",
        "Will the price of Bitcoin be between $97000 and $99000 on May 9?",
        "Will the price of Bitcoin be between $99000 and $101000 on May 9?",
        "Will the price of Bitcoin be between $101000 and $103000 on May 9?",
        "Will the price of Bitcoin be greater than $103000 on May 9?"
    ]
    cfg = default_cfg()
    res = answer_questions(
        QUESTIONS,
        ts_str="2025/5/4 14:36:57",
        csv_path=r"D:/prediction_market_data/crypto_data/merged_btc_feat.csv",
        cfg=cfg,
        DEBUG=True
    )
    print("\n============= RESULT =============")
    for q in QUESTIONS:
        print(f"{q:<65} → {res[q]:.2%}")
    print("90% interval:", res["pred_interval"])
