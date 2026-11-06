"""
Runs the predictors without skill (most frequent class, random guesses) on the same cross-environment splits, as the lowest reference point.

Needs: feature_temperature_damage_analysis.py.
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

sys.argv = [sys.argv[0]]            # make the imported module run in its default (full) mode
import feature_temperature_damage_analysis as fa

warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
N_SEEDS = 200
STRATEGIES = ["most_frequent", "stratified", "uniform"]


def predict(strategy, ytr, n_te, seed):
    """Predict with a dummy classifier fitted on the training labels."""
    m = DummyClassifier(strategy=strategy, random_state=seed)
    m.fit(np.zeros((len(ytr), 1)), ytr)
    return m.predict(np.zeros((n_te, 1)))


def cross_env(strategy, y, temp):
    """Macro F1 and accuracy of a dummy classifier on the cross-environment splits."""
    reps = 1 if strategy == "most_frequent" else N_SEEDS
    f1s, accs, n = [], [], 0
    for thr in fa.THRESHOLDS_C:
        low, high = np.where(temp < thr)[0], np.where(temp >= thr)[0]
        for tr, te in ((low, high), (high, low)):
            if len(tr) < 5 or len(te) < 5 or len(np.unique(y[tr])) < 2:
                continue
            n += 1
            fs, as_ = [], []
            for s in range(reps):
                f, a = fa.scores(y[te], predict(strategy, y[tr], len(te), s))
                fs.append(f)
                as_.append(a)
            f1s.append(np.mean(fs))
            accs.append(np.mean(as_))
    return float(np.mean(f1s)), float(np.mean(accs)), n


def plain_cv(strategy, y):
    """Macro F1 and accuracy of a dummy classifier under stratified 5-fold cross-validation."""
    reps = 1 if strategy == "most_frequent" else N_SEEDS
    fs, as_ = [], []
    for s in range(reps):
        pred = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros((len(y), 1)), y):
            pred[te] = predict(strategy, y[tr], len(te), s)
        f, a = fa.scores(y, pred)
        fs.append(f)
        as_.append(a)
    return float(np.mean(fs)), float(np.mean(as_))


def main():
    """Run the dummy classifiers on the cross-environment splits and plain cross-validation and save the table."""
    graphs, class_names = fa.load_graphs()
    xs, es, temp, y = fa.session_arrays(graphs)
    counts = pd.Series(y).map(dict(enumerate(class_names))).value_counts()
    print(f"\n{len(graphs)} sessions, {len(class_names)} classes; largest class "
          f"{counts.index[0]} = {counts.iloc[0] / len(y):.3f}\n")
    rows = []
    for st in STRATEGIES:
        f_c, a_c, n = cross_env(st, y, temp)
        f_p, a_p = plain_cv(st, y)
        rows.append(dict(predictor=st, cross_env_macro_f1=f_c, cross_env_accuracy=a_c, n_settings=n,
                         random_cv_macro_f1=f_p, random_cv_accuracy=a_p))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(HERE, "dummy_baselines.csv"), index=False)
    print(f"\nSaved {os.path.join(HERE, 'dummy_baselines.csv')}")


if __name__ == "__main__":
    main()
