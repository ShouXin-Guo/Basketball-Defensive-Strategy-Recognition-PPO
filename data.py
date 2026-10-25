"""Load annotated event features without inventing defensive strategy labels."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

FEATURES = [f'F{i}' for i in range(1, 12)]


@dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def load_events(path):
    """Read an event-level CSV (F1..F11,label,game_id) or NPZ (x,y,game_id).

    x has shape (events, frames, 11), or (events,11) for single-frame summaries.
    F1..F4 are scalar movement measurements for ordered players O1,O2,D1,D2.
    F5 is midpoint-to-basket distance; F6..F11 are other-player distances.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Dataset not found: {p}')
    if p.suffix.lower() == '.csv':
        df = pd.read_csv(p)
        needed = FEATURES + ['label', 'game_id']
        if any(c not in df for c in needed):
            raise ValueError(f'CSV must contain {needed}')
        x = df[FEATURES].to_numpy(dtype=np.float32)
        y = df['label'].to_numpy()
        groups = df['game_id'].astype(str).to_numpy()
    elif p.suffix.lower() == '.npz':
        with np.load(p, allow_pickle=False) as arr:
            if not {'x', 'y', 'game_id'}.issubset(arr.files):
                raise ValueError('NPZ requires x, y, game_id arrays')
            x = arr['x'].astype(np.float32)
            y = arr['y']
            groups = arr['game_id'].astype(str)
    else:
        raise ValueError('Supported input formats are .csv and .npz')
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or x.shape[2] != 11 or x.shape[1] < 1:
        raise ValueError('Expected features [events,frames,11]')
    y = np.asarray([str(z).strip().lower() for z in y])
    mapper = {'switch': 0, 'trap': 1, '0': 0, '1': 1}
    if not set(y).issubset(mapper):
        raise ValueError('Labels must be Switch/Trap or 0/1')
    y = np.array([mapper[z] for z in y], dtype=np.int64)
    if len(x) != len(y) or len(y) != len(groups) or not np.isfinite(x).all():
        raise ValueError('Inconsistent arrays or non-finite features')
    if len(set(groups)) < 3 or len(set(y)) != 2:
        raise ValueError('Require at least three games and both classes')
    return Events(x, y, np.asarray(groups))


def split_events(data, seed, test_fraction):
    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    tr, te = next(split.split(data.x, data.y, data.groups))
    if len(np.unique(data.y[tr])) != 2 or len(np.unique(data.y[te])) != 2:
        raise ValueError('Group hold-out failed to preserve both classes; adjust group structure or seed')
    return tr, te


def folds_for(data, indices, n_splits, seed):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for t, v in splitter.split(data.x[indices], data.y[indices], data.groups[indices]):
        if len(np.unique(data.y[indices[t]])) < 2 or len(np.unique(data.y[indices[v]])) < 2:
            raise ValueError('A grouped fold contains only one class')
        yield indices[t], indices[v]


def standardize(data, train_idx, eval_idx):
    """Fit scaler on training events/frames only and transform other events."""
    scaler = StandardScaler()
    xt = data.x[train_idx]
    scaler.fit(xt.reshape(-1, 11))
    def transform(ids):
        arr = data.x[ids]
        return scaler.transform(arr.reshape(-1, 11)).reshape(arr.shape).astype(np.float32)
    return transform(train_idx), transform(eval_idx), scaler
