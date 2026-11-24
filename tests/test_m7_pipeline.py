"""Synthetic regression tests for the M7 feature and validation pipeline."""

import sys
import tempfile
import unittest
import hashlib
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_ml.m7_pipeline import (
    EMBEDDING_DIM,
    build_interval_dataset,
    choose_f1_threshold,
    extract_interval_features,
    load_feature_cache,
    nested_site_loso,
    save_feature_cache,
    site_from_filename,
    species_support,
)


class FakeEncoder:
    def __init__(self, *, bad_shape=False, nonfinite=False):
        self.bad_shape = bad_shape
        self.nonfinite = nonfinite
        self.last_waveform = None

    def infer(self, waveform):
        self.last_waveform = waveform.copy()
        dim = 3 if self.bad_shape else EMBEDDING_DIM
        values = np.ones((2, dim), dtype=np.float32)
        if self.nonfinite:
            values[0, 0] = np.nan
        return np.zeros((2, 521), dtype=np.float32), values, np.zeros((2, 64), dtype=np.float32)


class FakeResampy:
    @staticmethod
    def resample(waveform, source_rate, target_rate):
        assert (source_rate, target_rate) == (32_000, 16_000)
        return waveform[::2]


def synthetic_loso_data():
    rows = []
    targets = []
    rng = np.random.default_rng(7)
    features = []
    for site in range(9):
        for item in range(10):
            label_a = int(site < 5 and item < 4)
            label_b = int(3 <= site < 8 and item >= 6)
            rows.append({
                "filename": f"BC2026_Train_{site:04d}_S{site:02d}_20250101_000000.ogg",
                "start_seconds": item * 5,
                "end_seconds": item * 5 + 5,
                "site_group": f"{site:02d}",
            })
            targets.append([label_a, label_b])
            features.append(rng.normal(size=EMBEDDING_DIM).astype(np.float32) + label_a * 0.5 - label_b * 0.25)
    return np.asarray(features), np.asarray(targets, dtype=np.uint8), pd.DataFrame(rows)


class M7PipelineTests(unittest.TestCase):
    def test_site_group_comes_from_authoritative_filename_pattern(self):
        self.assertEqual(site_from_filename("BC2026_Train_0001_S08_20250606_030007.ogg"), "08")
        with self.assertRaises(ValueError):
            site_from_filename("bad_filename.ogg")

    def test_support_rule_requires_intervals_recordings_and_sites(self):
        windows = pd.DataFrame({
            "filename": [f"f{i}.ogg" for i in range(25)] + ["rare.ogg"],
            "site_group": [f"s{i % 5}" for i in range(25)] + ["s1"],
        })
        targets = np.zeros((26, 2), dtype=np.uint8)
        targets[:20, 0] = 1
        targets[25, 1] = 1
        support = species_support(windows, targets, ("supported", "rare")).set_index("primary_label")
        self.assertTrue(bool(support.loc["supported", "eligible"]))
        self.assertFalse(bool(support.loc["rare", "eligible"]))

    def test_exact_interval_slicing_excludes_adjacent_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "BC2026_Train_0001_S01_20250101_000000.wav"
            signal = np.concatenate([np.zeros(160_000), np.full(160_000, 0.25), np.zeros(1_600_000)]).astype(np.float32)
            sf.write(audio, signal, 32_000, subtype="PCM_16")
            windows = pd.DataFrame([{
                "filename": audio.name, "start_seconds": 5, "end_seconds": 10, "site_group": "01"
            }])
            encoder = FakeEncoder()
            with patch.dict(sys.modules, {"resampy": FakeResampy}):
                features, details = extract_interval_features(windows, root, encoder)
            self.assertEqual(features.shape, (1, EMBEDDING_DIM))
            self.assertEqual(encoder.last_waveform.shape, (80_000,))
            self.assertTrue(np.allclose(encoder.last_waveform, encoder.last_waveform[0], atol=0.001))
            self.assertAlmostEqual(float(encoder.last_waveform.mean()), 0.25, delta=0.002)
            self.assertEqual(int(details.iloc[0].yamnet_patch_count), 2)

    def test_invalid_feature_shapes_and_nonfinite_embeddings_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "BC2026_Train_0001_S01_20250101_000000.wav"
            sf.write(audio, np.zeros(1_920_000, dtype=np.float32), 32_000, subtype="PCM_16")
            windows = pd.DataFrame([{"filename": audio.name, "start_seconds": 0, "end_seconds": 5, "site_group": "01"}])
            for encoder in (FakeEncoder(bad_shape=True), FakeEncoder(nonfinite=True)):
                with patch.dict(sys.modules, {"resampy": FakeResampy}):
                    with self.assertRaises(ValueError):
                        extract_interval_features(windows, root, encoder)

    def test_cache_roundtrip_and_mismatch_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "features.npz"
            metadata = root / "features.json"
            rows = pd.DataFrame({"filename": ["x.ogg"], "start_seconds": [0]})
            rows["end_seconds"] = 5
            matrix = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
            meta = {"source_hash": "abc", "version": 1, "interval_keys_sha256": hashlib.sha256(b"x.ogg|0|5").hexdigest()}
            save_feature_cache(cache, metadata, matrix, rows, meta)
            loaded, loaded_rows = load_feature_cache(cache, metadata, meta, 1)
            np.testing.assert_array_equal(loaded, matrix)
            self.assertEqual(loaded_rows.filename.tolist(), ["x.ogg"])
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                load_feature_cache(cache, metadata, {**meta, "source_hash": "different"}, 1)
            rows.loc[0, "filename"] = "wrong.ogg"
            rows.to_csv(root / "interval_feature_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "mapping"):
                load_feature_cache(cache, metadata, meta, 1)

    def test_threshold_selection_requires_both_classes_and_is_bounded(self):
        y = np.asarray([0, 0, 1, 1])
        p = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold = choose_f1_threshold(y, p)
        self.assertGreaterEqual(threshold, 0.05)
        self.assertLessEqual(threshold, 0.95)
        with self.assertRaises(ValueError):
            choose_f1_threshold(np.ones(4), p)

    def test_nested_site_loso_is_group_disjoint_and_reproducible(self):
        X, y, windows = synthetic_loso_data()
        first = nested_site_loso(X, y, windows, ("label_a", "label_b"))
        second = nested_site_loso(X, y, windows, ("label_a", "label_b"))
        self.assertEqual(first["eligible_labels"], ["label_a", "label_b"])
        self.assertEqual(len(first["folds"]), 9)
        for fold in first["folds"]:
            self.assertEqual(fold["validation_sites"], 1)
        for fold, frame in first["split_manifest"].groupby("outer_fold"):
            self.assertEqual(frame.loc[frame.partition.eq("validation"), "site_group"].nunique(), 1)
            self.assertEqual(frame.loc[frame.partition.eq("train"), "site_group"].nunique(), 8)
            self.assertFalse(set(frame.loc[frame.partition.eq("train"), "site_group"]) & set(frame.loc[frame.partition.eq("validation"), "site_group"]))
        np.testing.assert_allclose(first["probabilities"], second["probabilities"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_outer_validation_labels_do_not_select_that_folds_threshold(self):
        X, y, windows = synthetic_loso_data()
        baseline = nested_site_loso(X, y, windows, ("label_a", "label_b"))
        changed = y.copy()
        site8 = windows.site_group.eq("08").to_numpy()
        changed[site8, 0] = 1 - changed[site8, 0]
        altered = nested_site_loso(X, changed, windows, ("label_a", "label_b"))
        baseline_fold = next(row for row in baseline["fold_thresholds"] if row["held_out_site"] == "08")
        altered_fold = next(row for row in altered["fold_thresholds"] if row["held_out_site"] == "08")
        self.assertEqual(baseline_fold["label_a"], altered_fold["label_a"])

    def test_unsupported_species_are_excluded_and_zero_positive_site_metrics_are_undefined(self):
        X, y, windows = synthetic_loso_data()
        targets = np.column_stack([y, np.zeros(len(y), dtype=np.uint8)])
        result = nested_site_loso(X, targets, windows, ("label_a", "label_b", "rare"))
        self.assertEqual(result["eligible_labels"], ["label_a", "label_b"])
        absent_site = result["per_site"].query("site_group == '08' and primary_label == 'label_a'").iloc[0]
        self.assertEqual(absent_site.positive_support, 0)
        self.assertTrue(pd.isna(absent_site.precision))
        self.assertIn("no positive support", absent_site.undefined_reason)


if __name__ == "__main__":
    unittest.main()
