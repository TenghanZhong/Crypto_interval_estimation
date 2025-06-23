
"""
BTC Interval-Engine ★ Backtest Edition v4 – Δ-log-price with TFT + Early Stopping
-------------------------------------------------------
目标 = 未来 H 日 log-return；网络输出收益分位点；推断时平移回价格。
"""
import pytorch_lightning as pl, pytorch_forecasting as pf
print(pl.__version__, pf.__version__)  # 两者都应 ≥2.x

import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from pytorch_forecasting.data.encoders import NaNLabelEncoder
# PyTorch Forecasting / Lightning
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

warnings.filterwarnings("ignore")
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ————————— 参数配置 —————————
PRICE_WINDOW = 90

def default_cfg():
    return dict(
        train_fr=0.7,
        seq_len=60,
        epochs=300,
        batch=64,
        hidden=256,
        dropout=0.2,
        conf_lvl=0.90,
        n_sim=100000,
        quant_lo=0.05,
        quant_hi=0.95,
        quant_step=0.005,
        train_window_days=720,
        horizon=5,
        early_stop_patience=10,
        lr_reduce_patience=5,
        lr_reduce_factor = 0.5,
    )

# ————————— 工具函数 —————————

def build_quants(cfg):
    return np.round(
        np.arange(cfg['quant_lo'], cfg['quant_hi'] + 1e-9, cfg['quant_step']), 5
    )


def make_feature_matrix(df):
    out = df.copy()
    if 'iv' in out and out['iv'].max() > 2:
        out['iv'] /= 100
    if 'vol_spike_10' in out:
        out['vol_spike_10'] = np.log1p(out['vol_spike_10'])
    return out


def generate_intervals(p, step=2000):
    base_left = (int(p) // step) * step
    base_right = base_left + step
    bounds = [
        base_left - 3*step,
        base_left - 2*step,
        base_left - step,
        base_left,
        base_right,
        base_right + step,
    ]
    return [
        (-np.inf, bounds[0]),
        (bounds[0], bounds[1]),
        (bounds[1], bounds[2]),
        (bounds[2], bounds[3]),
        (bounds[3], bounds[4]),
        (bounds[4], bounds[5]),
        (bounds[5], np.inf),
    ]


def qloss(pred, targ, quants):
    q = torch.as_tensor(quants, device=pred.device)
    e = targ.unsqueeze(1) - pred
    return torch.mean(torch.maximum((q - 1) * e, q * e))

# ————————— 模型训练：TFT + Early Stopping —————————

def train_model(X, Y, tr_mask, cfg, device):
    """
    训练 TFT 模型并返回 (model, trainer, dataset_train)。
    """
    n_samples, L, n_feat = X.shape
    records = []

    # 1) 构造 encoder (t=0…L-1, y=0) + decoder (t=L, y=真值)
    for i in range(n_samples):
        for t in range(L):
            rec = {f"feat_{j}": X[i, t, j] for j in range(n_feat)}
            rec["time_idx"] = t
            rec["sample"]   = str(i)
            rec["y"]        = 0.0
            records.append(rec)
        dec = {f"feat_{j}": X[i, -1, j] for j in range(n_feat)}
        dec["time_idx"] = L
        dec["sample"]   = str(i)
        dec["y"]        = float(Y[i])
        records.append(dec)

    df_all = pd.DataFrame(records)

    # 2) 划分 train/val
    sample_int = df_all["sample"].astype(int)
    train_df = df_all[sample_int < tr_mask.sum()].copy()
    val_df   = df_all[sample_int >= tr_mask.sum()].copy()

    # 3) 构造 TimeSeriesDataSet
    dataset_train = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="y",
        group_ids=["sample"],
        min_encoder_length=L,
        max_encoder_length=L,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["sample"],
        time_varying_unknown_reals=[f"feat_{j}" for j in range(n_feat)],
        target_normalizer=GroupNormalizer(groups=["sample"]),
        categorical_encoders={"sample": NaNLabelEncoder(add_nan=True)},
    )
    dataset_val = TimeSeriesDataSet.from_dataset(
        dataset_train,
        val_df,
        stop_randomization=True,
        categorical_encoders={"sample": NaNLabelEncoder(add_nan=True)},
    )

    # 4) DataLoader
    loader_train = dataset_train.to_dataloader(train=True, batch_size=cfg["batch"], num_workers=0)
    loader_val   = dataset_val.to_dataloader(train=False, batch_size=cfg["batch"], num_workers=0)

    # 5) 构建 TFT（不要 .to(device)）
    quants = build_quants(cfg)
    tft = TemporalFusionTransformer.from_dataset(
        dataset_train,
        learning_rate=3e-4,
        hidden_size=cfg["hidden"],
        lstm_layers=2,
        dropout=cfg["dropout"],
        output_size=len(quants),
        loss=QuantileLoss(),
    )

    # 6) Trainer：只保留 EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=cfg["early_stop_patience"], mode="min")
    trainer = Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1 if device == "cuda" else None,
        callbacks=[early_stop],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar = False,  # ← 关闭 Lightning 进度条
    )

    trainer.fit(tft, train_dataloaders=loader_train, val_dataloaders=loader_val)
    return tft, trainer, dataset_train
# ————————— CQR 校准（改进版，不用 DataLoader） —————————
# ————————— CQR 校准 —————————
# ————————— CQR 校准 —————————
def calibrate_cqr(mdl, dataset_train, X_cal, Y_cal, quants, conf_lvl, cfg):
    mdl.eval()
    n_cal, L, n_feat = X_cal.shape
    # 1) 只拼 decoder 步 (t=L)的校准集
    records = []
    for i in range(n_cal):
        rec = {f"feat_{j}": X_cal[i, -1, j] for j in range(n_feat)}
        rec["time_idx"] = L
        rec["sample"]   = str(i)
        rec["y"]        = float(Y_cal[i])
        records.append(rec)
    df_cal = pd.DataFrame(records)
    # 2) from_dataset(predict=True)
    dataset_cal = TimeSeriesDataSet.from_dataset(
        dataset_train,
        df_cal,
        predict=True,
        stop_randomization=True,
    )
    # 3) 直接 mdl.predict 拿到 (n_cal, n_quants)
    preds = mdl.predict(dataset_cal, mode="quantiles")  # np.ndarray
    # 4) 计算 δ_lo, δ_hi
    err_lo = np.maximum(Y_cal - preds[:, 0], 0)
    err_hi = np.maximum(preds[:, -1] - Y_cal, 0)
    k = int(np.ceil((len(err_lo) + 1) * (1 - (1 - conf_lvl) / 2))) - 1
    return np.sort(err_lo)[k], np.sort(err_hi)[k]


# ————————— 区间推断 —————————
# ————————— 区间推断 —————————
def infer_intervals(mdl, dataset_train, win_pred_z, δ_lo, δ_hi, quants, cfg, last_price):
    mdl.eval()
    L, n_feat = win_pred_z.shape
    records = []
    # encoder 部分
    for t in range(L):
        rec = {f"feat_{j}": win_pred_z[t, j] for j in range(n_feat)}
        rec["time_idx"] = t
        rec["sample"]   = "0"
        records.append(rec)
    # decoder 最后一步
    dec = {f"feat_{j}": win_pred_z[-1, j] for j in range(n_feat)}
    dec["time_idx"] = L
    dec["sample"]   = "0"
    records.append(dec)
    df_inf = pd.DataFrame(records)
    dataset_inf = TimeSeriesDataSet.from_dataset(
        dataset_train,
        df_inf,
        predict=True,
        stop_randomization=True,
    )
    preds = mdl.predict(dataset_inf, mode="quantiles")[0]
    # 累积 & 校准
    q_r = np.maximum.accumulate(preds)
    q_r[0]  -= δ_lo
    q_r[-1] += δ_hi
    q_r = np.maximum.accumulate(q_r)
    # 模拟生成概率
    ln_pt = np.log(last_price)
    q_lp  = ln_pt + q_r
    u     = np.random.rand(cfg["n_sim"])
    idx   = np.clip(np.searchsorted(quants, u) - 1, 0, len(quants)-2)
    sims_lp = q_lp[idx] + (u-quants[idx])*(q_lp[idx+1]-q_lp[idx])/cfg["quant_step"]
    sims_price = np.exp(sims_lp)
    intervals = generate_intervals(last_price)
    return intervals, [((sims_price>=lo)&(sims_price<=hi)).mean() for lo,hi in intervals]

# ————————— 命中评估 —————————
import torch
def evaluate_hits(intervals, probs, true_price):
    order = np.argsort(probs)[::-1]
    best, second = order[0], order[1]
    hit_b = intervals[best][0] <= true_price <= intervals[best][1]
    hit_s = intervals[second][0] <= true_price <= intervals[second][1]
    true_bin = next(i for i, (lo, hi) in enumerate(intervals) if lo <= true_price <= hi)
    return hit_b, hit_s, probs[true_bin]

# ————————— 回测主流程 —————————

def backtest(csv_path, cfg, verbose=True, rolling_cols=None, static_cols=None, raw_cols=None):
    rolling_cols = rolling_cols or []
    static_cols  = static_cols or []
    raw_cols     = raw_cols or []

    df = (pd.read_csv(csv_path, parse_dates=['date'])
            .rename(columns=str.strip).rename(columns=str.lower)
            .set_index('date').sort_index())
    df = make_feature_matrix(df)

    L, H = cfg['seq_len'], cfg['horizon']
    look = cfg['train_window_days'] + L + PRICE_WINDOW + H
    start, end = look - 1, len(df) - 1 - H

    hit_b_tot = hit_s_tot = total = prob20_cnt=0
    prob_sum = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quants = build_quants(cfg)

    for i in trange(start, end + 1, desc='Rolling', ncols=70):
        seg = df.iloc[i-look+1:i+1]
        base = seg.iloc[:-H]
        mu = base[rolling_cols].rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW).mean().reindex(seg.index, method='ffill')
        sd = base[rolling_cols].rolling(PRICE_WINDOW, min_periods=PRICE_WINDOW).std().add(1e-12).reindex(seg.index, method='ffill')
        for c in rolling_cols:
            seg[f'{c}_z'] = (seg[c] - mu[c]) / sd[c]
        feat = seg[[f'{c}_z' for c in rolling_cols] + static_cols + raw_cols].dropna()

        price = seg['log_price']
        df_y = feat.iloc[:-H].copy()
        df_y['y'] = (price.shift(-H) - price).reindex(df_y.index)

        df_y.dropna(subset=['y'], inplace=True)
        split = int(len(df_y) * cfg['train_fr'])

        μ_s = df_y[static_cols].iloc[:split].mean()
        σ_s = df_y[static_cols].iloc[:split].std().add(1e-12)
        df_y[static_cols] = (df_y[static_cols] - μ_s) / σ_s
        feat[static_cols] = (feat[static_cols] - μ_s) / σ_s

        feature_cols = [c for c in df_y.columns if c != 'y']
        X, Y = [], []
        for j in range(L, len(df_y)):
            X.append(df_y[feature_cols].iloc[j-L:j].values)
            Y.append(df_y['y'].iloc[j])
        X = np.array(X)
        Y = np.array(Y)

        tr_mask = np.arange(len(X)) < (split - L)

        mdl, trainer, dataset_train = train_model(X, Y, tr_mask, cfg, device)
        X_cal, Y_cal = X[~tr_mask], Y[~tr_mask]
        δ_lo, δ_hi = calibrate_cqr(mdl, dataset_train, X_cal, Y_cal, quants, cfg['conf_lvl'], cfg)
        win_z  = feat.tail(L).values
        last_p = np.exp(seg['log_price'].iloc[-1])
        ints, ps = infer_intervals(mdl, dataset_train, win_z, δ_lo, δ_hi, quants, cfg, last_p)
        true_p = np.exp(df['log_price'].iloc[i + cfg['horizon']])
        hb, hs, pc = evaluate_hits(ints, ps, true_p)

        hit_b_tot += hb
        hit_s_tot += hs
        total    += 1
        prob_sum += pc

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
    print(f"Days: {total}, Hit_best: {hit_b_tot}({hit_b_tot / total:.2%}), "
          f"Hit_second: {hit_s_tot}({hit_s_tot / total:.2%}), "
          f"Avg_prob_correct: {prob_sum / total:.2%}),"
          f"Prob≥20% but not in First/Second Interval: {prob20_cnt}({prob20_cnt / total:.2%})")


# ————————— 运行 —————————
if __name__ == '__main__':
    cfg = default_cfg()
    path = r"D:\prediction_market_data\crypto_data\merged_btc3.csv"
    rolling_cols = ['log_price', 'hv5', 'iv']
    static_cols  = ['oc_ret', 'ema5_10', 'vol_spike_10', 'fear_index']
    raw_cols     = ['ret1_z', 'ret5_z', 'rsi14_z']
    backtest(path, cfg, verbose=True, rolling_cols=rolling_cols,
             static_cols=static_cols, raw_cols=raw_cols)

