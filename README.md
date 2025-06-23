# Crypto_interval_estimation
# 🧠 BTC Interval-Engine

## 🎯 Objective

This engine uses **LSTM-based quantile regression** combined with **Conformalized Quantile Regression (CQR)** to dynamically forecast the future \( H \)-day log-return distribution of Bitcoin. The model generates discrete price intervals with associated probabilities and evaluates hit rates and confidence calibration.

---

## 📁 Input Data Format

The input must be a CSV file containing the following columns:

- **`date`**: Trading date in `YYYY-MM-DD` format  
- **`log_price`**: Natural logarithm of the closing price  
- **`hv5`**: 5-day historical volatility  
- **`iv`**: Implied volatility (divide by 100 if values > 2)  
- **`oc_ret`**: Open-to-close return  
- **`ema5_10`**: Difference between 5-day and 10-day EMA  
- **`vol_spike_10`**: 10-day volume spike indicator  
- **`fear_index`**: Market fear index  
- **`ret1_z`**: Z-score of 1-day return  
- **`ret5_z`**: Z-score of 5-day return  
- **`rsi14_z`**: Z-score of 14-day RSI

---

## 🔍 Preprocessing

- **Datetime parsing**: Convert `date` to datetime index and sort in ascending order.  
- **Volume spike transformation**: Apply \( \ln(1 + x) \) to `vol_spike_10`.

---

## 🏗️ Feature Engineering

### Rolling Standardization (Z-score)

- Features: `log_price`, `hv5`, `iv`  
- Window: `PRICE_WINDOW = 90` days  
- Apply rolling mean and std, then z-score. Fill missing values forward.

### Static Standardization

- Features: `oc_ret`, `ema5_10`, `vol_spike_10`, `fear_index`  
- Split window into training (70%) and validation (30%)  
- Compute mean/std on training only, then normalize entire window.

### Raw Features

- `ret1_z`, `ret5_z`, `rsi14_z`: Directly appended without transformation.

---

## 🪟 Sliding Window & Sample Construction

### Key Parameters

- `train_window_days = 720`: Historical lookback period  
- `seq_len = 60`: LSTM input sequence length  
- `PRICE_WINDOW = 90`: Rolling normalization window  
- `horizon = 3`: Predict 3-day future log-return  

### Lookback Length

\[
\text{look} = 720 + 60 + 90 + 3 = 873 \text{ days}
\]

- Start index: `872`  
- End index: `len(df) - 1 - horizon`  

### For Each Window

- Extract range `[i - 872, i]`  
- Apply feature construction  
- Drop rows with missing values  
- Label:
  \[
  y_t = \ln P_{t + \text{horizon}} - \ln P_t
  \]

---

## 🧪 Dataset Splitting (per window)

- Total samples: \( M = 873 - 60 - 3 = 810 \)  
- **Training set**: First \( \lfloor 0.7M \rfloor \) samples (e.g., 567)  
- **Validation set**: Remaining samples (e.g., 243)  
- Used for EarlyStopping, LR scheduling, and CQR calibration

---

## 🧠 Model Training

### Architecture

- 2-layer LSTM  
- LayerNorm  
- 2-layer MLP head  
- Output: 19 quantile values

### Quantile Range

\[
[0.05, 0.95] \text{ with step } 0.005
\]

### Loss Function

- **Pinball Loss**: averaged over quantiles

### Optimization

- Optimizer: `AdamW`, learning rate = `3e-4`  
- LR scheduler: `ReduceLROnPlateau`, patience=5, factor=0.5  
- EarlyStopping: patience=10

---

## 📏 Conformal Quantile Calibration (CQR)

After training, the model outputs predicted quantiles on the validation set: `q_hat_0.05`, `q_hat_0.10`, ..., `q_hat_0.95`.
To calibrate the prediction intervals using Conformal Quantile Regression (CQR), we first compute residuals for each validation sample: `e_lo = max(y - q_hat_0.05, 0)` and `e_hi = max(q_hat_0.95 - y, 0)`. 
These residuals represent how much the true value exceeds the lower bound or falls short of the upper bound. 
Given a desired confidence level `alpha = 0.90`, we then select the `k`-th smallest values from `e_lo` and `e_hi`, where `k = floor((1 - alpha) * (n + 1))`, and use them as calibration margins: `delta_lo` and `delta_hi`. 
Finally, we adjust the predicted quantiles to form the calibrated interval: `q_cal_0.05 = q_hat_0.05 - delta_lo` and `q_cal_0.95 = q_hat_0.95 + delta_hi`, so that the resulting interval `[q_cal_0.05, q_cal_0.95]` achieves approximately 90% empirical coverage on unseen data.

---

## 🔮 Interval Inference & Evaluation

### Conformal Calibration

To generate the final calibrated prediction interval, we apply the error margins obtained from the validation set to the model's original quantiles:  
`q_cal_0.05 = q_hat_0.05 - delta_lo`  
`q_cal_0.95 = q_hat_0.95 + delta_hi`  
This produces the calibrated interval `[q_cal_0.05, q_cal_0.95]`, which guarantees a marginal coverage rate no less than the target confidence level (e.g., 90%) under finite samples.

### Inverse-Transform Sampling

We draw `n_sim` samples `u_i ~ Uniform(0, 1)` from a uniform distribution.  
Given the calibrated quantile levels `{q_j}` and their corresponding log-price values `{l_j}`, each `u_i` is mapped via linear interpolation:  
`l_tilde_i = Interp(u_i; (q_j, l_j))`  
`P_tilde_i = exp(l_tilde_i)`  
The resulting samples `{P_tilde_i}` follow the model's predicted conditional distribution over future prices.

### Discrete Price Interval Construction

Take the latest observed price `P_last` as the center, and define 7 consecutive price intervals using a fixed step size `Delta = $2000`:  
`(-∞, P_last - 3Δ)`, `(P_last - 3Δ, P_last - 2Δ)`, ..., `(P_last + 2Δ, P_last + 3Δ)`, `(P_last + 3Δ, ∞)`

### Probability Calculation and Interval Selection

For each interval `k`, compute the proportion of simulated prices falling within that range:  
`p_k = (1 / n_sim) * sum_{i=1}^{n_sim} 1(P_tilde_i in interval_k)`  
Then, select the top-2 intervals with the highest probabilities and label them as `best` and `second`.

### Evaluation Metrics

- `hit_best`: Whether the true future price `P_true` falls into the highest-probability interval  
- `hit_second`: Whether `P_true` falls into the second-highest-probability interval  
- `prob_correct`: The predicted probability of the interval containing `P_true`

---

## ⚙️ Default Configuration

```python
def default_cfg():
    return dict(
        train_fr=0.7,
        seq_len=60,
        epochs=150,
        batch=64,
        hidden=256,
        dropout=0.2,
        conf_lvl=0.90,
        n_sim=100000,
        quant_lo=0.05,
        quant_hi=0.95,
        quant_step=0.005,
        train_window_days=720,
        horizon=3,
        PRICE_WINDOW=90,
    )
