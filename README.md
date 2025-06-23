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

Using predictions \( \hat{q}_{0.05}, ..., \hat{q}_{0.95} \) on the validation set:

Residuals:

\[
e_{lo} = \max(y - \hat{q}_{0.05}, 0), \quad
e_{hi} = \max(\hat{q}_{0.95} - y, 0)
\]

At confidence level \( \alpha = 0.90 \), select the \( k \)-th smallest residuals:

\[
\delta_{lo},\ \delta_{hi}
\]

---

## 🔮 Interval Inference & Evaluation

### Calibrated Intervals

\[
\hat{q}_{0.05}' = \hat{q}_{0.05} - \delta_{lo}, \quad
\hat{q}_{0.95}' = \hat{q}_{0.95} + \delta_{hi}
\]

### Monte Carlo Simulation

- Sample `n_sim = 100000` quantiles in [0.05, 0.95]  
- Interpolate to log-return, shift to log-price, apply `exp` to get price  
- Discretize into 7 intervals with \$2000 step size  

### Probability Assignment

- Count percentage of simulated prices falling in each interval  
- Select top-2 most probable intervals: `best`, `second`

### Evaluation Metrics

- `hit_best`: whether true price falls into the most probable interval  
- `hit_second`: falls into the second most probable  
- `prob_correct`: the predicted probability of the true price interval

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
