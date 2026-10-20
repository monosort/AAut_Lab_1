import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from build_lagged_features import build_lagged_frame, DEFAULT_LAGS

ALPHA = 1100
TARGET = "chlorophyll_a_mg_m3"

raw = pd.read_csv("chlorophyll_student_2015_2023.csv", parse_dates=["date"])
raw = raw.sort_values("date").reset_index(drop=True)
table = build_lagged_frame(raw, DEFAULT_LAGS)   # todas as colunas candidatas

def cv_score(cols):
    """R2 médio a validar em 2021, 2022 e 2023."""
    scores = []
    for val_year in [2021, 2022, 2023]:
        tr = table["date"].dt.year < val_year
        va = table["date"].dt.year == val_year
        Xtr, Xva = table.loc[tr, cols], table.loc[va, cols]
        mu, sd = Xtr.mean(), Xtr.std(ddof=0)
        m = Ridge(ALPHA).fit((Xtr - mu) / sd, table.loc[tr, TARGET])
        scores.append(r2_score(table.loc[va, TARGET], m.predict((Xva - mu) / sd)))
    return np.mean(scores)

candidates = [c for c in table.columns if c not in ("date", TARGET)]
selected, best = [], -np.inf

while True:
    scores = {c: cv_score(selected + [c]) for c in candidates if c not in selected}
    c, s = max(scores.items(), key=lambda kv: kv[1])
    if s < best + 0.001:
        break
    selected.append(c)
    best = s
    print(f"+ {c:35s} R2 médio = {s:.4f}")

print("\nColunas escolhidas:", selected)