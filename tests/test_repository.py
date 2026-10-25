"""Unit and integration checks independent of the private annotated dataset."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from config import Settings, save_config, load_config, list_configurations
from data import load_events, split_events, standardize
from models import ActorCritic
from ppo import fit, predict_prob, gae
from experiment import run, select_device, evaluate_checkpoint


class RepoTests(unittest.TestCase):
    def test_configuration_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'initial.json'
            save_config(Settings(), path)
            self.assertEqual(load_config(path), Settings())
            self.assertIn('initial.json', list_configurations(tmp))

    def test_training_and_checkpoint(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.csv'
            rows = []
            for g in range(20):
                for label in ['Switch', 'Trap']:
                    for _ in range(2):
                        features = rng.normal(size=11)
                        features[0] += (label == 'Trap')*0.5
                        rows.append(dict(zip([f'F{i}' for i in range(1, 12)], features),
                                         label=label, game_id=f'g{g}'))
            pd.DataFrame(rows).to_csv(path, index=False)
            data = load_events(path)
            tr, te = split_events(data, 42, .2)
            self.assertFalse(set(data.groups[tr]) & set(data.groups[te]))
            x, v, _ = standardize(data, tr, te)
            cfg = Settings(episodes=2, ppo_epochs=1, minibatch=8, policy_layers=2, policy_hidden=16)
            out = Path(tmp)/'out'
            reports = run(data, cfg, select_device('cpu'), out)
            self.assertEqual(len(reports), 1)
            self.assertTrue((out/'model_seed_42.pt').exists())
            restored = evaluate_checkpoint(data, out/'model_seed_42.pt', select_device('cpu'), out)
            self.assertAlmostEqual(restored['accuracy'], reports[0]['accuracy'])

    def test_gae_terminal(self):
        a, r = gae([1., 2.], [0., 0.], [False, True], 1., 1.)
        np.testing.assert_allclose(a, [3., 2.])

if __name__ == '__main__':
    unittest.main()
