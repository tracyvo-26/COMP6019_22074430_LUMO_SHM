"""
Helper that chooses the two loss weights on a validation part of the training data. Used by tune_lambda_backward.py.

Needs: compare_vanilla_cal.py.
"""
from sklearn.model_selection import train_test_split

VAL_FRACTION = 0.15


def make_chooser(m, grid):
    """Return a choose_lambda function that picks the loss weights on a validation split."""
    def choose_lambda(cfg, train_graphs, seed, node_dim, n_cls):
        """Pick the best loss weight pair for a CAL variant on a validation split of the training graphs."""
        if not (cfg["kind"] == "cal" and cfg["causal"]):
            return m.LAMBDA1, m.LAMBDA2
        labels = [g.y.item() for g in train_graphs]
        idx = list(range(len(train_graphs)))
        try:
            tr, va = train_test_split(idx, test_size=VAL_FRACTION, stratify=labels, random_state=seed)
        except ValueError:
            tr, va = train_test_split(idx, test_size=VAL_FRACTION, random_state=seed)
        tr_g, va_g = [train_graphs[i] for i in tr], [train_graphs[i] for i in va]
        best, best_score = (m.LAMBDA1, m.LAMBDA2), -1.0
        for l1, l2 in grid:
            r = m.fit_and_eval(cfg, tr_g, {"val": va_g}, seed, node_dim, n_cls, l1, l2)
            score = r["val_acc"] + r["val_f1"]
            if score > best_score:
                best, best_score = (l1, l2), score
        return best
    return choose_lambda
