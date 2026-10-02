#!/usr/bin/env python3

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np
import torch

import gpu_calorimeter_energy as benchmark


class EnergyBenchmarkTests(unittest.TestCase):
    def test_tiny_training_run_writes_reloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "geometry.xml"
            xml.write_text(
                '<Bins><Bin name="photon">'
                '<Layer id="0" r_edges="0,1" n_bin_alpha="1" />'
                '<Layer id="1" r_edges="0,1" n_bin_alpha="1" />'
                '<Layer id="2" r_edges="0,1" n_bin_alpha="1" />'
                '<Layer id="3" r_edges="0,1" n_bin_alpha="1" />'
                '<Layer id="4" r_edges="0,1" n_bin_alpha="1" />'
                '</Bin></Bins>',
                encoding="utf-8",
            )
            data = root / "events.hdf5"
            rng = np.random.default_rng(11)
            incidents = np.linspace(100.0, 1000.0, 24, dtype=np.float32)[:, None]
            showers = rng.gamma(2.0, 3.0, size=(24, 5)).astype(np.float32)
            with h5py.File(data, "w") as handle:
                handle.create_dataset("incident_energies", data=incidents)
                handle.create_dataset("showers", data=showers)
            output = root / "output"
            args = benchmark.argparse.Namespace(
                output_dir=output,
                train_file=data,
                geometry=xml,
                seed=17,
                train_size=16,
                validation_size=8,
                iterations=2,
                batch_size=4,
                validation_batch_size=8,
                validation_every=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                hidden_dim=16,
                hidden_layers=2,
                mixture_components=2,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            result = benchmark.run(args)
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "model.pt").exists())
            self.assertTrue((output / "best_model_diagnostic.pt").exists())
            self.assertTrue((output / "validation_samples.npz").exists())
            self.assertTrue(np.isfinite(result["best_validation_nll"]))
            self.assertTrue(np.isfinite(result["final_validation_nll"]))
            self.assertEqual(result["checkpoint_selection"], "final_step")
            self.assertEqual(len(result["history"]), 2)
            resumed = benchmark.run(args)
            self.assertEqual(resumed["checkpoint_selection"], "final_step")
            self.assertEqual(len(resumed["history"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
