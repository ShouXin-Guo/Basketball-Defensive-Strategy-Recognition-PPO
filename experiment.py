"""Grouped hold-out evaluation, grouped CV random search, and durable checkpoints."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from config import save_config
from data import split_events, folds_for, standardize
from evaluation import score
from models import ActorCritic
from ppo import fit, predict_prob


def select_device(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(request)


def candidate(base, rng):
    """Uniform/random search over reported Table 1 search space (not a claimed paper search algorithm)."""
    c = copy.deepcopy(base)
    c.episodes = int(rng.integers(32, 1025))
    c.branches = int(rng.integers(1, 6))
    c.policy_layers = int(rng.integers(2, 65))
    c.policy_hidden = int(rng.integers(16, 1025))
    c.value_layers = int(rng.integers(1, 6))
    c.value_hidden = int(rng.integers(8, 129))
    c.activation = str(rng.choice(['relu', 'leaky_relu', 'prelu', 'tanh', 'sigmoid']))
    c.reward_lambda = float(rng.uniform(0.05, .95))
    c.learning_rate = float(10**rng.uniform(-5, -3))
    c.optimizer = str(rng.choice(['adam','adamw','rmsprop','sgd']))
    c.clip_epsilon = float(rng.uniform(.05, .30))
    c.gamma = float(rng.uniform(.90, .999))
    c.gae_lambda = float(rng.uniform(.90, 1.0))
    c.ppo_epochs = int(rng.integers(1, 21))
    c.minibatch = int(rng.integers(8, 129))
    return c.validate()


def optimize(data, train_ids, cfg, device, out):
    rng = np.random.default_rng(cfg.seed)
    log = []
    best = None
    for trial in range(cfg.search_trials):
        trial_cfg = candidate(cfg, rng)
        fold_scores = []
        for fold, (t, v) in enumerate(folds_for(data, train_ids, cfg.folds, cfg.seed)):
            xt, xv, _ = standardize(data, t, v)
            model, _ = fit(xt, data.y[t], trial_cfg, device, cfg.seed + 1000*trial + fold)
            prob = predict_prob(model, xv, device)
            fold_scores.append(float(f1_score(data.y[v], prob>=.5, average='macro', zero_division=0)))
        result = {'trial': trial, 'macro_f1': float(np.mean(fold_scores)), 'folds': fold_scores,
                  'configuration': asdict(trial_cfg)}
        log.append(result)
        if best is None or result['macro_f1'] > best['macro_f1']:
            best = result
    out.mkdir(parents=True, exist_ok=True)
    (out/'search_results.json').write_text(json.dumps(log, indent=2))
    winner = type(cfg)(**best['configuration'])
    winner.status = 'optimized'
    save_config(winner, out/'optimized.json')
    return winner


def run(data, cfg, device, output, *, optimize_first=False, seeds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tr, te = split_events(data, cfg.seed, cfg.test_fraction)
    if optimize_first:
        cfg = optimize(data, tr, cfg, device, output)
    xt, xe, scaler = standardize(data, tr, te)
    seeds = seeds or [cfg.seed]
    reports = []
    all_pred = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, history = fit(xt, data.y[tr], cfg, device, seed)
        prob = predict_prob(model, xe, device)
        report = {'seed': seed, **score(data.y[te], prob)}
        reports.append(report)
        all_pred.append(prob)
        ckpt_path = output/f'model_seed_{seed}.pt'
        torch.save({'state_dict': model.cpu().state_dict(), 'config': asdict(cfg),
                    'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                    'seed': seed}, ckpt_path)
        (output/f'history_seed_{seed}.json').write_text(json.dumps(history, indent=2))
    cfg.status = 'trained'
    save_config(cfg, output/'trained_settings.json', overwrite=True)
    (output/'metrics.json').write_text(json.dumps(reports, indent=2))
    np.savez_compressed(output/'test_predictions.npz', labels=data.y[te],
                        probabilities=np.asarray(all_pred), game_id=data.groups[te])
    return reports


def evaluate_checkpoint(data, checkpoint_path, device, output):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError('A real trained checkpoint is required for evaluation')
    from config import Settings
    cfg = Settings(**ckpt['config']).validate()
    model = ActorCritic(cfg).to(device)
    model.load_state_dict(ckpt['state_dict'])
    _, te = split_events(data, cfg.seed, cfg.test_fraction)
    arr = data.x[te]
    mean = np.asarray(ckpt['scaler_mean'])
    std = np.asarray(ckpt['scaler_scale'])
    standardized = ((arr-mean)/std).astype('float32')
    report = score(data.y[te], predict_prob(model, standardized, device))
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/'checkpoint_metrics.json').write_text(json.dumps(report, indent=2))
    return report
