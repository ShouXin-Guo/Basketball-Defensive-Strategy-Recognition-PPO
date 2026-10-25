"""Metrics and paired game-level bootstrap."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def score(y, probabilities):
    yhat = (probabilities >= .5).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {'accuracy': float(accuracy_score(y, yhat)),
            'f_measure_macro': float(f1_score(y, yhat, average='macro', zero_division=0)),
            'f_measure_trap': float(f1_score(y, yhat, pos_label=1, zero_division=0)),
            'g_means': float(np.sqrt(np.prod(recalls))),
            'auc': float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else None,
            'confusion_matrix': cm.tolist()}


def paired_game_bootstrap(y, proposed, baseline, groups, metric='f_measure_macro', n=2000, seed=42):
    """Paired cluster bootstrap; report percentile CI, not a null-centered p-value."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    samples = []
    for _ in range(n):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        samples.append(score(y[idx], proposed[idx])[metric] - score(y[idx], baseline[idx])[metric])
    if not samples:
        raise ValueError('Bootstrap samples lack both classes')
    return {'observed_delta': score(y, proposed)[metric] - score(y, baseline)[metric],
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'bootstrap_samples': len(samples)}
