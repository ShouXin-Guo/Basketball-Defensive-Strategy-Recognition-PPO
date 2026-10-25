"""Early-termination, minority-reward PPO with generalized advantage estimation."""
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from models import ActorCritic


def make_optimizer(model, cfg):
    kinds = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
             'rmsprop': torch.optim.RMSprop, 'sgd': torch.optim.SGD}
    return kinds[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)


def gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) and not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        advantages[t] = last
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def train_ppo(x, y, cfg, device, *, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    model = ActorCritic(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    x = torch.from_numpy(x).to(device)
    history = []
    for episode in range(cfg.episodes):
        model.eval()
        order = rng.permutation(len(y))
        buffer = []
        with torch.no_grad():
            for index in order:
                state = x[index:index+1]
                logits, v = model(state)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                a = int(action.item())
                label = int(y[index])
                magnitude = 1.0 if label == 1 else cfg.reward_lambda
                reward = magnitude if a == label else -magnitude
                done = bool(label == 1 and a != label) or index == order[-1]
                buffer.append((int(index), a, float(distribution.log_prob(action).item()),
                               float(v.item()), reward, done))
                if done:
                    break
        if not buffer:
            raise RuntimeError('Empty episode')
        indices, actions, old_logp, values, rewards, dones = zip(*buffer)
        advantages, returns = gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = torch.tensor(advantages, device=device)
        ret = torch.tensor(returns, device=device)
        old = torch.tensor(old_logp, device=device)
        acts = torch.tensor(actions, device=device)
        for _ in range(cfg.ppo_epochs):
            for mb in np.array_split(rng.permutation(len(buffer)), max(1, int(np.ceil(len(buffer) / cfg.minibatch)))):
                if len(mb) == 0:
                    continue
                batch_idx = torch.as_tensor(mb, device=device)
                logits, val = model(x[np.asarray(indices)[mb]])
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(acts[batch_idx]) - old[batch_idx])
                a = adv[batch_idx]
                objective = torch.minimum(ratio*a, torch.clamp(ratio, 1-cfg.clip_epsilon, 1+cfg.clip_epsilon)*a)
                loss = -objective.mean() + cfg.value_coef*F.mse_loss(val, ret[batch_idx]) - cfg.entropy_coef*dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
        history.append({'episode': episode+1, 'steps': len(buffer), 'reward': float(np.sum(rewards))})
    return model, history


def train_supervised(x, y, cfg, device, *, seed=None):
    """No-PPO ablation uses a conventional cross-entropy classifier."""
    if seed is not None:
        torch.manual_seed(seed)
    model = ActorCritic(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    x = torch.from_numpy(x).to(device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(cfg.episodes):
        losses = []
        for ids in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y)/cfg.minibatch)))):
            logits, _ = model(x[ids])
            loss = F.cross_entropy(logits, y[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append({'episode': epoch+1, 'loss': float(np.mean(losses))})
    return model, history


def fit(x, y, cfg, device, seed):
    if cfg.model_variant == 'no_ppo':
        return train_supervised(x, y, cfg, device, seed=seed)
    return train_ppo(x, y, cfg, device, seed=seed)


def predict_prob(model, x, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/batch_size)))):
            logits, _ = model(torch.from_numpy(chunk).to(device))
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)
