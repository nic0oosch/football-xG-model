# Expected Goals (xG) Model

A machine learning project that predicts the probability of a shot resulting in a goal (xG) using spatial freeze-frame data from StatsBomb Open Data.

Developed as a bachelor's learning / fun project exploring whether **spatial context** from freeze-frames improves xG prediction over classical feature-based approaches.

---

## Results

All models evaluated on a held-out validation set. Test set is reserved for final thesis evaluation.

| Model | AUC | Brier |
|---|---|---|
| Logistic Regression (Baseline) | 0.712 | 0.0783 |
| XGBoost | 0.758 | 0.0762 |
| XGNet (heatmap only) | 0.749 | 0.0961 |
| HybridXGNet (CNN + tabular) | 0.768 | 0.0905 |
| **Stacked (XGNet → XGBoost)** | **0.776** | **0.0917** |
| StatsBomb xG *(reference)* | 0.792 | 0.0739 |

> AUC: higher is better (max 1.0) · Brier: lower is better (min 0.0)

---

## Architecture

### Model Hierarchy

```
Logistic Regression       distance + angle only
        ↓
XGBoost                   + body part, shot type, defender count, goalkeeper distance
        ↓
XGNet                     + freeze-frame heatmap (CNN)
        ↓
HybridXGNet               CNN branch + tabular MLP branch (end-to-end)
        ↓
Stacked (XGNet → XGBoost) CNN embeddings + tabular → XGBoost classifier
```

### Freeze-Frame Heatmap

Each shot is encoded as a `(3 × 68 × 52)` heatmap:

```
Channel 0  →  Teammate positions
Channel 1  →  Opponent positions
Channel 2  →  Shooter position
```

Gaussian blur is applied to each channel to reduce sparsity.

### HybridXGNet

```
Heatmap (3×68×52) ──► CNN Branch (3×Conv+BN+ReLU+Pool) ──► 1536-dim
                                                                  │
                                                             cat([·, ·])
                                                                  │
Tabular Features  ──► MLP Branch (Linear → ReLU → Dropout) ──►  32-dim
                                                                  │
                                                          Linear(1568, 64)
                                                                  │
                                                           Dropout(0.5)
                                                                  │
                                                           Linear(64, 1)
                                                                  │
                                                                 xG
```

### Stacked Model

Trained XGNet is used as a **feature extractor** (final classification layer removed → 64-dim embedding). Embeddings are concatenated with tabular features and fed into XGBoost.

---

## Data

Uses [StatsBomb Open Data](https://github.com/statsbomb/open-data), free for non-commercial and educational use.

**Competitions included:**

| Competition | Season            |
|---|-------------------|
| UEFA Euro | 2020, 2024        |
| FIFA World Cup | 2018, 2022        |
| 1. Bundesliga | 2015/16, 2023/24  |
| La Liga | 2017/18 – 2020/21 |
| Champions League | 2016/17 – 2018/19 |
| AFCON | 2023              |

**Data is not included in this repository** due to StatsBomb's terms of use. It is fetched automatically via the `statsbombpy` API on first run and cached locally under `data/processed/`.

---

## Project Structure

```
football-xG-model/
├── src/
│   ├── data.py                  # loading, caching, train/val/test split
│   ├── features.py              # feature engineering, freeze-frame → heatmap
│   ├── evaluate.py              # AUC, Brier Score, StatsBomb comparison
│   └── models/
│       ├── __init__.py
│       ├── base.py              # shared preprocessor, make_X_* functions
│       ├── baseline.py          # Logistic Regression
│       ├── xgboost.py           # XGBoost
│       ├── cnn_models.py        # XGNet, HybridXGNet architectures
│       ├── cnn_training.py      # training loops, early stopping, LR scheduler
│       ├── cnn_dataset.py       # PyTorch ShotDataset
│       └── stacked_training.py  # XGNetEmbedder, train_stacked, extract_embeddings
└── notebooks/
    ├── 01_exploration.ipynb           # data exploration, freeze-frame visualization
    ├── 02_model_sanity_check.ipynb    # pipeline integrity checks
    ├── 03_XGBoost_vs_Baseline.ipynb   # tabular model comparison
    └── 04_CNN_heatmap_tabular.ipynb   # CNN models, full comparison
```

---

## Setup

```bash
git clone https://github.com/nic0oosch/football-xG-model.git
cd football-xG-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Data is fetched automatically on first run:

```bash
cd notebooks
jupyter notebook 01_exploration.ipynb
```

---

## Requirements

- Python >= 3.12
- PyTorch
- XGBoost
- scikit-learn
- statsbombpy
- pandas / numpy / scipy
- matplotlib
- pyarrow

Full pinned dependencies in `requirements.txt`.

---

## Development

This project was developed in phases, each tracked as a separate branch and documented via PR descriptions that serve as a running research log.

```
main                  stable releases only
└── dev               current combined state
      ├── phase0/baseline     Logistic Regression, Euro 2024 only
      ├── phase1/xgboost      extended dataset, XGBoost, extended features
      └── phase2/cnn          XGNet, HybridXGNet, Stacked model
```

---

## License

MIT © 2026 Nico

*StatsBomb Open Data is subject to [StatsBomb's terms of use](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) and is not included in this repository.*
