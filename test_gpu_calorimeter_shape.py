#!/usr/bin/env python3

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np
import torch

import calorimeter_energy as energy
import gpu_calorimeter_shape as benchmark


class ShapeBenchmarkTests(unittest.TestCase):
    def test_tiny_training_run_writes_reloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "geometry.xml"
            xml.write_text(
                '<Bins><Bin name="photon">'
                + "".join(
                    f'<Layer id="{layer}" r_edges="0,1,2" n_bin_alpha="1" />'
                    for layer in range(5)
                )
                + "</Bin></Bins>",
                encoding="utf-8",
            )
            rng = np.random.default_rng(19)
            incidents = np.linspace(100.0, 2000.0, 24, dtype=np.float32)
            showers = rng.gamma(2.0, 3.0, size=(24, 10)).astype(np.float32)
            data = root / "events.hdf5"
            with h5py.File(data, "w") as handle:
                handle.create_dataset("incident_energies", data=incidents[:, None])
                handle.create_dataset("showers", data=showers)
            layer_energies = showers.reshape(24, 5, 2).sum(axis=-1)
            transform = energy.fit_energy_transform(incidents[:16], layer_energies[:16], range(5))
            energy_checkpoint = root / "energy.pt"
            torch.save({"energy_transform": transform.to_dict()}, energy_checkpoint)
            output = root / "output"
            args = benchmark.argparse.Namespace(
                output_dir=output,
                train_file=data,
                geometry=xml,
                energy_checkpoint=energy_checkpoint,
                seed=23,
                train_size=16,
                validation_size=8,
                iterations=2,
                batch_size=4,
                validation_batch_size=4,
                validation_every=1,
                validation_samples=4,
                learning_rate=1e-3,
                weight_decay=0.0,
                core_components=3,
                hidden_dim=12,
                transformer_layers=1,
                attention_heads=3,
                mixture_components=2,
                residual_hidden_dim=16,
                residual_layers=3,
                residual_loss_weight=1.0,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            result = benchmark.run(args)
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "model.pt").exists())
            self.assertTrue((output / "best_model_diagnostic.pt").exists())
            self.assertTrue((output / "validation_shape_samples.npz").exists())
            self.assertEqual(result["checkpoint_selection"], "final_step")
            self.assertEqual(len(result["history"]), 2)
            self.assertTrue(np.isfinite(result["final_validation"]["loss"]))
            resumed = benchmark.run(args)
            self.assertEqual(len(resumed["history"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
