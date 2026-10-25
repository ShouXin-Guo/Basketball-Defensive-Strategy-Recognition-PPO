"""Validated experiment configuration and durable JSON persistence."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    status: str = 'initialized'
    seed: int = 42
    episodes: int = 200
    branches: int = 4
    policy_layers: int = 4
    policy_hidden: int = 25
    value_layers: int = 2
    value_hidden: int = 64
    activation: str = 'relu'
    reward_lambda: float = 0.4
    learning_rate: float = 0.0003
    optimizer: str = 'adam'
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    minibatch: int = 32
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    folds: int = 5
    test_fraction: float = 0.2
    search_trials: int = 12
    model_variant: str = 'proposed'

    def validate(self):
        assert self.status in {'initialized', 'optimized', 'trained'}
        assert self.activation in {'relu', 'leaky_relu', 'prelu', 'tanh', 'sigmoid'}
        assert self.optimizer in {'adam', 'adamw', 'rmsprop', 'sgd'}
        assert self.model_variant in {'proposed', 'no_cnn', 'no_ppo'}
        for key in ('episodes', 'branches', 'policy_layers', 'policy_hidden', 'value_layers', 'value_hidden', 'ppo_epochs', 'minibatch', 'folds', 'search_trials'):
            assert type(getattr(self, key)) is int and getattr(self, key) > 0, key
        assert 1 <= self.branches <= 5
        assert 0 < self.reward_lambda < 1
        assert 0 < self.clip_epsilon < 1
        assert 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
        assert 0 < self.test_fraction < 0.5 and self.folds >= 2
        assert self.learning_rate > 0 and self.value_coef >= 0 and self.entropy_coef >= 0
        return self


def load_config(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {f.name for f in fields(Settings)}
    if not isinstance(obj, dict) or set(obj) != expected:
        raise ValueError(f'Configuration keys mismatch: missing={expected-set(obj)}, unexpected={set(obj)-expected}')
    return Settings(**obj).validate()


def save_config(config, path, *, overwrite=False):
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Configuration already exists: {path}')
    path.write_text(json.dumps(asdict(config), indent=2) + '\n', encoding='utf-8')
    return path


def list_configurations(directory):
    paths = sorted(Path(directory).glob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No JSON configurations in {directory}')
    return {p.name: load_config(p) for p in paths}
