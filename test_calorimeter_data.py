#!/usr/bin/env python3

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np
import torch

import calorimeter_data as calo
import gpu_model as gm


class CalorimeterDataTests(unittest.TestCase):
    @staticmethod
    def make_fixture(root: Path):
        xml = root / "geometry.xml"
        xml.write_text(
            '<Bins><Bin name="photon"><Layer id="0" r_edges="0,1,3" n_bin_alpha="2" />'
            '<Layer id="2" r_edges="0,2" n_bin_alpha="1" /></Bin></Bins>', encoding="utf-8"
        )
        geometry = calo.load_geometry(xml)
        data = root / "events.hdf5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset("incident_energies", data=np.asarray([[10.0], [20.0]], dtype=np.float32))
            handle.create_dataset(
                "showers",
                data=np.asarray([[1.0, 2.0, 3.0, 0.0, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            )
        return data, geometry

    def test_geometry_and_conservative_topk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, geometry = self.make_fixture(root)
            self.assertEqual(geometry.voxel_count, 5)
            records = calo.load_records(
                data, geometry, count=2, max_components=2, relative_threshold=0.0,
                log_energy_mean=np.log(10.0), log_energy_std=1.0,
            )
            self.assertEqual([record.state.n for record in records], [2, 2])
            self.assertTrue(all(record.state.conservation_residual() < 1e-7 for record in records))
            self.assertAlmostEqual(float(records[0].state.masses.sum()), 0.5, places=6)
            self.assertEqual(records[0].state.feature_dim, 3)

    def test_dense_protocol_round_trip_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, geometry = self.make_fixture(root)
            dense = calo.load_dense_batch(data, count=2, min_energy_mev=1.0)
            np.testing.assert_array_equal(
                dense.showers_mev,
                np.asarray([[1.0, 2.0, 3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            state = gm.WeightedSetState(
                torch.tensor([2.0, 0.5, 1.0]),
                torch.tensor(geometry.features[[1, 1, 4]]),
                0.5,
                4.0,
                (1, 1, 4),
            )
            rasterized = calo.states_to_dense_showers([state], geometry, mass_scales_mev=np.asarray([2.0]))
            np.testing.assert_allclose(rasterized, [[0.0, 5.0, 0.0, 0.0, 2.0]])
            output = root / "sample.hdf5"
            calo.write_official_hdf5(output, rasterized, np.asarray([10.0]))
            reloaded = calo.load_dense_batch(output)
            np.testing.assert_allclose(reloaded.showers_mev, rasterized)
            np.testing.assert_allclose(reloaded.incident_energies_mev, [10.0])

    def test_dataset1_incident_schedule(self):
        first = calo.dataset1_incident_energies(repeats=1)
        self.assertEqual(first.shape, (121, 1))
        self.assertEqual(int(np.count_nonzero(first == 2.0**8)), 10)
        self.assertEqual(int(np.count_nonzero(first == 2.0**19)), 5)
        self.assertEqual(int(np.count_nonzero(first == 2.0**22)), 1)
        shuffled = calo.dataset1_incident_energies(repeats=2, seed=7)
        self.assertEqual(shuffled.shape, (242, 1))
        np.testing.assert_array_equal(np.sort(shuffled[:, 0]), np.sort(np.tile(first[:, 0], 2)))

    def test_dense_export_rejects_off_grid_features(self):
        with tempfile.TemporaryDirectory() as directory:
            _, geometry = self.make_fixture(Path(directory))
            state = gm.make_state([1.0], [[9.0, 9.0, 9.0]], reservoir=0.0)
            with self.assertRaises(ValueError):
                calo.states_to_dense_showers([state], geometry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
