"""Testa features de interacao (a*b) e quadraticas (a^2) e combinacoes delas.

Para cada candidata (e para combinacoes de candidatas) treina um modelo com
as features base + as novas e devolve o R2 medio em varios anos de validacao.

Correr a partir de training_code/:
    python select_interactions.py --data chlorophyll_student_2015_2023.csv
    python select_interactions.py --data chlorophyll_student_2015_2023.csv --all_pairs
"""

from __future__ import annotations

import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score

from build_lagged_features import DEFAULT_LAGS, build_lagged_frame

TARGET = "chlorophyll_a_mg_m3"

# Features base: o teu lag_spec atual (muda aqui se mudares no train.py)
BASE_LAG_SPEC = {
    "chlorophyll_a_mg_m3": list(range(1, 9)),
    "sst_c": list(range(0, 8)),
    "par_umol_m2_s": list(range(0, 8)),
    "nitrate_umol_l": [3],
    "wind_speed_m_s": list(range(0, 5)),
    "upwelling_index": [2],
    "mixed_layer_depth_m": list(range(0, 8)),
    "current_speed_m_s": list(range(0, 8)),
    "river_discharge_index": list(range(0, 8)),
    "cloud_fraction": [0],
    "surface_pressure_hpa": list(range(0, 8)),
    "turbidity_ntu": [5],
}

# Candidatas da tua tabela: (coluna_a, coluna_b). Se a == b e quadrado.
TABLE_CANDIDATES = [
    ("nitrate_umol_l__lag_3", "par_umol_m2_s__lag_1"),
    ("upwelling_index__lag_2", "par_umol_m2_s__lag_1"),
    ("upwelling_index__lag_2", "upwelling_index__lag_2"),
    ("sst_c__lag_1", "nitrate_umol_l__lag_3"),
    ("par_umol_m2_s__lag_1", "mixed_layer_depth_m__lag_1"),
    ("nitrate_umol_l__lag_3", "nitrate_umol_l__lag_3"),
    ("par_umol_m2_s__lag_1", "par_umol_m2_s__lag_1"),
    ("upwelling_index__lag_2", "current_speed_m_s__lag_0"),
    ("par_umol_m2_s__lag_1", "turbidity_ntu__lag_1"),
]

# Todas as colunas dos lags default (131 colunas) -> usadas em --all_pairs
DEFAULT_COLUMNS = [f"{v}__lag_{l}" for v, ls in DEFAULT_LAGS.items() for l in ls]

# Colunas "chave" usadas no modo --auto (todos os quadrados e todos os pares)
AUTO_KEY_COLUMNS = [
    "chlorophyll_a_mg_m3__lag_1",
    "sst_c__lag_1",
    "par_umol_m2_s__lag_1",
    "nitrate_umol_l__lag_3",
    "upwelling_index__lag_2",
    "wind_speed_m_s__lag_1",
    "mixed_layer_depth_m__lag_1",
    "current_speed_m_s__lag_0",
    "turbidity_ntu__lag_1",
    "river_discharge_index__lag_1",
]


def feature_name(a: str, b: str) -> str:
    return f"{a}^2" if a == b else f"{a}*{b}"


def variable_of(col: str) -> str:
    return re.sub(r"__lag_\d+$", "", col)


def required_lag_spec(pairs) -> dict[str, list[int]]:
    """Base + os lags que as candidatas precisam (para as colunas existirem)."""
    spec = {k: set(v) for k, v in BASE_LAG_SPEC.items()}
    for pair in pairs:
        for col in pair:
            var, lag = re.fullmatch(r"(.+)__lag_(\d+)", col).groups()
            spec.setdefault(var, set()).add(int(lag))
    return {k: sorted(v) for k, v in spec.items()}


def make_scorer(table: pd.DataFrame, base_cols: list[str], extra: dict[str, np.ndarray],
                estimator: str, alpha: float, epsilon: float,
                years: list[int], eval_years: list[int]):
    """Devolve (cv_score, per_year).

    cv_score(nomes)  -> R2 medio nos anos de SELECAO (--years)
    per_year(nomes)  -> {ano: R2} para cada ano de AVALIACAO (--eval_years)
    Em ambos, para validar o ano V treina-se com todos os anos anteriores a V.
    """
    year = table["date"].dt.year
    y = table[TARGET].to_numpy()
    base_X = table[base_cols].to_numpy(dtype=float)
    folds = {}
    for v in sorted(set(years) | set(eval_years)):
        tr, va = (year < v).to_numpy(), (year == v).to_numpy()
        if va.sum() == 0 or tr.sum() < 365:
            raise SystemExit(f"ano {v}: sem dados de validacao ou menos de 1 ano de treino")
        folds[v] = (tr, va)

    def design(extra_names):
        if extra_names:
            return np.column_stack([base_X] + [extra[n] for n in extra_names])
        return base_X

    def fold_r2(X, v) -> float:
        tr, va = folds[v]
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0, ddof=0)
        sd[sd == 0] = 1.0  # evita divisao por zero
        if estimator == "huber":
            model = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=20000)
        else:
            model = Ridge(alpha)
        model.fit((X[tr] - mu) / sd, y[tr])
        return float(r2_score(y[va], model.predict((X[va] - mu) / sd)))

    def cv_score(extra_names) -> float:
        X = design(extra_names)
        return float(np.mean([fold_r2(X, v) for v in years]))

    def per_year(extra_names) -> dict[int, float]:
        X = design(extra_names)
        return {v: fold_r2(X, v) for v in eval_years}

    return cv_score, per_year


def score_all(names, cv_score, label: str):
    """Pontua cada candidata sozinha, com progresso e estimativa de tempo."""
    out, start, step = [], time.time(), max(1, len(names) // 10)
    for i, n in enumerate(names, 1):
        out.append((cv_score([n]), (n,)))
        if i % step == 0:
            elapsed = time.time() - start
            print(f"  [{label}] {i}/{len(names)}  ({elapsed:.0f}s, faltam ~{elapsed / i * (len(names) - i):.0f}s)",
                  flush=True)
    return sorted(out, reverse=True)


def show(title: str, rows: list[tuple[float, tuple[str, ...]]], baseline: float, n: int):
    print(f"\n=== {title} ===")
    print(f"{'R2 medio':>9} {'delta':>8}  features adicionadas")
    for score, feats in rows[:n]:
        print(f"{score:9.4f} {score - baseline:+8.4f}  {' + '.join(feats)}")


def robustness_report(sets, per_year, eval_years, sel_years, show_n: int, names):
    """Avalia os melhores conjuntos ano a ano e por janelas de N anos."""
    sets = [("BASE", ())] + [(f"S{i}", f) for i, f in enumerate(sets, 1)]
    res = {label: per_year(list(feats)) for label, feats in sets}
    base = res["BASE"]
    years_desc = sorted(eval_years, reverse=True)
    held = [v for v in eval_years if v not in sel_years]
    mark = lambda v: f"{v}{'*' if v in sel_years else ' '}"

    print("\n\n##### ROBUSTEZ POR ANO #####")
    print(f"(* = ano usado para escolher as candidatas; os outros sao anos 'novos': {held})")
    print("\nLegenda dos conjuntos:")
    for label, feats in sets[1:]:
        print(f"  {label}: {' + '.join(feats)}")

    print("\n--- R2 em cada ano de validacao (treino = todos os anos anteriores) ---")
    print(f"{'':5}" + "".join(f"{mark(v):>8}" for v in sorted(eval_years)))
    for label, _ in sets:
        print(f"{label:5}" + "".join(f"{res[label][v]:8.4f}" for v in sorted(eval_years)))

    print("\n--- delta medio vs BASE nos ultimos N anos de validacao ---")
    print(f"{'':5}" + "".join(f"{'ult'+str(k):>8}" for k in range(1, len(years_desc) + 1)) + f"{'anos>BASE':>11}")
    for label, _ in sets[1:]:
        cells = []
        for k in range(1, len(years_desc) + 1):
            last = years_desc[:k]
            cells.append(np.mean([res[label][v] - base[v] for v in last]))
        wins = sum(res[label][v] > base[v] for v in eval_years)
        print(f"{label:5}" + "".join(f"{c:+8.4f}" for c in cells) + f"{wins:>7}/{len(eval_years)}")

    print(f"\n--- RANKING: melhores conjuntos pelo delta nos anos NOVOS {held} ---")
    if not held:
        print("Todos os --eval_years foram usados na selecao; usa --eval_years com anos fora de --years.")
        ranked = sorted(sets[1:], key=lambda it: -np.mean([res[it[0]][v] - base[v] for v in eval_years]))
        held_for_rank = eval_years
    else:
        ranked = sorted(sets[1:], key=lambda it: -np.mean([res[it[0]][v] - base[v] for v in held]))
        held_for_rank = held
    print(f"{'conj':5}{'delta anos selecao':>20}{'delta anos novos':>18}{'anos>BASE (novos)':>19}  features")
    for label, feats in ranked[:show_n]:
        d_sel = np.mean([res[label][v] - base[v] for v in sel_years if v in res[label]] or [np.nan])
        d_new = np.mean([res[label][v] - base[v] for v in held_for_rank])
        wins = sum(res[label][v] > base[v] for v in held_for_rank)
        print(f"{label:5}{d_sel:+20.4f}{d_new:+18.4f}{wins:>14}/{len(held_for_rank)}  {' + '.join(feats)}")
    print("\nSe o delta nos anos novos for muito menor que nos anos de selecao, o conjunto estava sobreajustado.")
    best = ranked[0][1] if ranked else ()
    if best:
        print("\nMelhor conjunto (anos novos) -> linhas para o train.py:")
        for f in best:
            a, b = names[f]
            print(f'  X["{f}"] = X["{a}"] * X["{b}"]')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--estimator", choices=["ridge", "huber"], default="ridge")
    parser.add_argument("--alpha", type=float, default=None,
                        help="default: 1100 (ridge) ou 0.01 (huber)")
    parser.add_argument("--epsilon", type=float, default=1.75, help="so para huber")
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021],
                        help="anos de validacao usados para ESCOLHER as candidatas "
                             "(treina sempre com os anos anteriores)")
    parser.add_argument("--eval_years", type=int, nargs="+",
                        default=[2018, 2019, 2020, 2021, 2022, 2023],
                        help="anos usados na analise final de robustez (inclui anos fora de --years)")
    parser.add_argument("--eval_top", type=int, default=30,
                        help="quantos dos melhores conjuntos entram na analise de robustez")
    parser.add_argument("--auto", action="store_true",
                        help="testa tambem todos os quadrados e pares de AUTO_KEY_COLUMNS")
    parser.add_argument("--all_pairs", action="store_true",
                        help="testa TODOS os quadrados e pares das 131 colunas dos lags default "
                             "(~8650 candidatas; usar com ridge)")
    parser.add_argument("--top_singles", type=int, default=10,
                        help="quantas candidatas individuais entram nas combinacoes")
    parser.add_argument("--diverse", action="store_true",
                        help="nas combinacoes, no maximo 1 candidata por par de variaveis "
                             "(evita escolher sst_lag1*chl_lag1 e sst_lag2*chl_lag1 juntas)")
    parser.add_argument("--max_combo", type=int, default=3,
                        help="tamanho maximo das combinacoes (2 a N, ex: 6)")
    parser.add_argument("--greedy_top", type=int, default=0,
                        help="selecao progressiva so sobre as K melhores candidatas "
                             "(0 = todas; recomendado ~300 com --all_pairs)")
    parser.add_argument("--show", type=int, default=10, help="quantos resultados mostrar")
    args = parser.parse_args()

    # 1. Lista de candidatas (sem repetidas)
    pairs = list(TABLE_CANDIDATES)
    if args.auto:
        pairs += list(itertools.combinations_with_replacement(AUTO_KEY_COLUMNS, 2))
    if args.all_pairs:
        pairs += list(itertools.combinations_with_replacement(DEFAULT_COLUMNS, 2))
    seen, unique = set(), []
    for a, b in pairs:
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            unique.append((a, b))
    pairs = unique

    # 2. Tabela com todos os lags necessarios + produtos candidatos (em numpy)
    raw = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    table = build_lagged_frame(raw, required_lag_spec(pairs))
    columns = {c: table[c].to_numpy(dtype=float) for c in table.columns if c not in ("date", TARGET)}
    extra, names = {}, {}
    for a, b in pairs:
        name = feature_name(a, b)
        extra[name] = (columns[a] * columns[b]).astype(np.float32)  # valores ORIGINAIS, antes de standardizar
        names[name] = (a, b)

    base_cols = [f"{v}__lag_{l}" for v, ls in BASE_LAG_SPEC.items() for l in ls]
    base_cols += ["annual_sin", "annual_cos"]
    alpha = args.alpha if args.alpha is not None else (0.01 if args.estimator == "huber" else 1100)
    cv_score, per_year = make_scorer(table, base_cols, extra, args.estimator, alpha,
                                     args.epsilon, args.years, args.eval_years)

    print(f"Estimador: {args.estimator} (alpha={alpha}"
          + (f", epsilon={args.epsilon}" if args.estimator == "huber" else "") + ")")
    baseline = cv_score([])
    print(f"Base: {len(base_cols)} colunas (incl. annual_sin/cos)")
    print(f"Baseline (sem features novas): R2 medio = {baseline:.4f}  "
          f"[selecao em {args.years}]")
    print(f"{len(names)} candidatas a testar...")

    # 3. Cada candidata sozinha
    singles = score_all(list(names), cv_score, "individuais")
    show("Candidatas individuais", singles, baseline, args.show)

    # 4. Combinacoes das melhores candidatas
    top, used_vars = [], set()
    for _, feats in singles:
        n = feats[0]
        key = frozenset(variable_of(c) for c in names[n])
        if args.diverse and key in used_vars:
            continue
        used_vars.add(key)
        top.append(n)
        if len(top) == args.top_singles:
            break
    combos = []
    for size in range(2, args.max_combo + 1):
        for combo in itertools.combinations(top, size):
            combos.append((cv_score(list(combo)), combo))
    combos.sort(reverse=True)
    show(f"Combinacoes (de {len(top)} melhores, tamanho 2 a {args.max_combo})",
         combos, baseline, args.show)

    # 5. Selecao progressiva
    pool = list(names) if args.greedy_top == 0 else [f[0] for _, f in singles[: args.greedy_top]]
    selected, best = [], baseline
    while True:
        remaining = [n for n in pool if n not in selected]
        if not remaining:
            break
        scores = {n: cv_score(selected + [n]) for n in remaining}
        n, s = max(scores.items(), key=lambda kv: kv[1])
        if s < best + 0.001:
            break
        selected.append(n)
        best = s
        print(f"  [progressiva] + {n}  R2 medio = {s:.4f}", flush=True)
    greedy = [(best, tuple(selected))] if selected else []
    show("Selecao progressiva (acrescenta enquanto melhorar >= 0.001)", greedy, baseline, 1)

    # 6. Top final (tudo junto)
    everything, seen_sets = [], set()
    for score, feats in sorted(singles + combos + greedy, reverse=True):
        if frozenset(feats) not in seen_sets:  # a progressiva pode repetir uma combinacao
            seen_sets.add(frozenset(feats))
            everything.append((score, feats))
    show(f"TOP {args.show} GERAL", everything, baseline, args.show)

    robustness_report([f for _, f in everything[: args.eval_top]], per_year,
                      args.eval_years, args.years, args.show, names)


if __name__ == "__main__":
    main()