"""Synthetic regression tests for the isolated BioSonic M5 YAMNet workflow."""

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_yamnet import config
from biosonic_yamnet.audio import AudioValidationError, prepare_waveform
from biosonic_yamnet.classifiers import _score_matrix, make_group_cv_splits, pipeline_and_grid
from biosonic_yamnet.dataset import (DatasetIntegrityError, aggregate_patch_embeddings,
                                     persist_embeddings, read_holdout_manifest_after_freeze,
                                     read_training_manifest)
from biosonic_yamnet.io_utils import sha256_file
from biosonic_yamnet.model import YamnetEncoder


class DummyScores:
    def __init__(self):
        self.classes_ = np.array(list(reversed(config.WORKING_LABELS)))

    def predict_proba(self, x):
        return np.tile(np.arange(len(self.classes_), dtype=np.float64), (len(x), 1))


def synthetic_metadata(repeats=10):
    rows = []
    for label in config.WORKING_LABELS:
        for collection in ("XC", "iNat"):
            for replicate in range(repeats):
                rows.append({
                    "filename": f"{label}/{collection}_{replicate}.ogg",
                    "primary_label": label, "collection": collection,
                    "group_id": f"g_{label}_{collection}_{replicate}",
                })
    return pd.DataFrame(rows)


class M5AudioTests(unittest.TestCase):
    def test_resample_synthetic_mono_32k_to_16k(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "synthetic.wav"
            sr = 32000
            t = np.arange(sr * 5, dtype=np.float32) / sr
            wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(path, wave, sr, subtype="FLOAT")
            out, info = prepare_waveform(path)
            self.assertEqual(out.shape, (80000,))
            self.assertEqual(info["model_sample_rate"], 16000)
            self.assertTrue(np.isfinite(out).all())
            self.assertEqual(info["sample_count"], 160000)

    def test_input_must_be_mono_32khz_and_finite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "wrong-rate.wav"
            sf.write(path, np.zeros(1000, dtype=np.float32), 16000)
            with self.assertRaises(AudioValidationError):
                prepare_waveform(path)
        encoder = object.__new__(YamnetEncoder)
        with self.assertRaises(ValueError):
            encoder.infer(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            encoder.infer(np.array([np.nan], dtype=np.float32))


class M5EmbeddingTests(unittest.TestCase):
    def test_recording_aggregation_is_arithmetic_mean(self):
        patches = np.vstack([np.zeros(config.EMBEDDING_DIM), np.ones(config.EMBEDDING_DIM) * 2])
        result = aggregate_patch_embeddings(patches)
        self.assertEqual(result.shape, (1024,))
        np.testing.assert_allclose(result, 1.0)

    def test_rejects_empty_wrong_shape_and_nonfinite_embeddings(self):
        for value in (np.empty((0, 1024)), np.zeros((2, 10)), np.full((2, 1024), np.inf)):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    aggregate_patch_embeddings(value)

    def test_persisted_label_and_filename_arrays_are_safe_unicode(self):
        frame = pd.DataFrame({
            "filename": ["burowl/XC1.ogg"], "primary_label": ["burowl"],
            "partition": ["train"],
        })
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            with patch.object(config, "DATA_ROOT", Path(temp_dir)):
                matrix_path, _ = persist_embeddings(
                    np.zeros((1, config.EMBEDDING_DIM), dtype=np.float32), frame,
                    "train", {"recordings": 1},
                )
                with np.load(matrix_path, allow_pickle=False) as saved:
                    self.assertEqual(saved["labels"].dtype.kind, "U")
                    self.assertEqual(saved["filenames"].dtype.kind, "U")

    def test_duplicate_recording_ids_rejected_in_cv(self):
        frame = synthetic_metadata(repeats=5)
        frame.loc[1, "filename"] = frame.loc[0, "filename"]
        with self.assertRaises(ValueError):
            make_group_cv_splits(frame)

    def test_grouped_cv_is_reproducible_and_leakage_safe(self):
        frame = synthetic_metadata(repeats=10)
        frame.loc[1, "group_id"] = frame.loc[0, "group_id"]
        frame.loc[2, "group_id"] = frame.loc[0, "group_id"]
        one = make_group_cv_splits(frame, n_splits=5, seed=42)
        two = make_group_cv_splits(frame, n_splits=5, seed=42)
        self.assertEqual(len(one), 5)
        for (train1, val1), (train2, val2) in zip(one, two):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)
            self.assertFalse(set(frame.group_id.iloc[train1]) & set(frame.group_id.iloc[val1]))
            self.assertEqual(set(frame.primary_label.iloc[val1]), set(config.WORKING_LABELS))

    def test_scalers_are_inside_only_scaled_classifier_pipelines(self):
        lr, _ = pipeline_and_grid("logistic_regression")
        svm, _ = pipeline_and_grid("svm")
        forest, _ = pipeline_and_grid("random_forest")
        self.assertEqual(lr.steps[0][0], "scaler")
        self.assertEqual(svm.steps[0][0], "scaler")
        self.assertNotIn("scaler", [name for name, _ in forest.steps])
        x = np.arange(48, dtype=np.float64).reshape(16, 3)
        y = np.repeat(np.array(list(config.WORKING_LABELS)), 2)
        lr.fit(x, y)
        np.testing.assert_allclose(lr.named_steps["scaler"].mean_, x.mean(axis=0))

    def test_score_columns_follow_fixed_class_order(self):
        x = np.zeros((2, 1))
        scores, kind = _score_matrix(DummyScores(), x)
        self.assertEqual(kind, "probability")
        original = list(reversed(config.WORKING_LABELS))
        expected = [original.index(label) for label in config.WORKING_LABELS]
        np.testing.assert_array_equal(scores[0], expected)


class M5PartitionTests(unittest.TestCase):
    def test_training_reader_stops_at_training_partition_boundary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "split.csv"
            fieldnames = ["filename", "primary_label", "collection", "group_id", "author_group_id",
                          "sha256", "expected_size_bytes", "actual_size_bytes", "sample_rate", "channels",
                          "sample_count", "duration_seconds", "partition", "unknown_author"]
            rows = []
            for index in range(config.TRAIN_RECORDINGS):
                label = config.WORKING_LABELS[index % len(config.WORKING_LABELS)]
                collection = ("XC", "iNat")[(index // len(config.WORKING_LABELS)) % 2]
                rows.append({"filename": f"{label}/r{index}.ogg", "primary_label": label,
                             "collection": collection, "group_id": f"g{index}",
                             "author_group_id": f"a{index}",
                             "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                             "expected_size_bytes": 1, "actual_size_bytes": 1,
                             "sample_rate": 32000, "channels": 1, "sample_count": 32000,
                             "duration_seconds": 1.0, "partition": "train", "unknown_author": False})
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow({"filename": "holdout/is_not_loaded.ogg", "partition": "validation"})
                stream.write("malformed row after partition boundary\n")
            loaded = read_training_manifest(path)
            self.assertEqual(len(loaded), config.TRAIN_RECORDINGS)
            self.assertTrue(loaded.partition.eq("train").all())
            self.assertNotIn("holdout/is_not_loaded.ogg", set(loaded.filename))

    def test_holdout_reader_requires_frozen_classifier_before_opening_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            bad_split = Path(temp_dir) / "must_not_be_read.csv"
            bad_split.write_text("not a csv", encoding="utf-8")
            with self.assertRaises(PermissionError):
                read_holdout_manifest_after_freeze(bad_split, Path(temp_dir) / "missing-freeze.json")

    def test_m2_sample_exclusion_constant_is_disjoint_from_working_labels(self):
        self.assertEqual(len(config.M2_SAMPLE_FILES), 6)
        self.assertEqual(len(config.WORKING_LABELS), 8)


class M5ArtifactTests(unittest.TestCase):
    def test_source_and_checkpoint_hashes_match_pins(self):
        for name, expected in config.OFFICIAL_SOURCE_SHA256.items():
            self.assertEqual(sha256_file(config.OFFICIAL_SOURCE_DIR / name), expected)
        self.assertEqual(sha256_file(config.WEIGHTS_PATH), config.WEIGHTS_SHA256)


if __name__ == "__main__":
    unittest.main()
