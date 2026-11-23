"""M6 optimization unit and leakage-boundary tests."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_ml import config
from biosonic_ml import m6_optimization as m6


def small_metadata() -> pd.DataFrame:
    rows = []
    for label in config.WORKING_LABELS:
        for collection in ("XC", "iNat"):
            for index in range(10):
                rows.append({"filename": f"{label}_{collection}_{index}.ogg",
                             "primary_label": label, "collection": collection,
                             "group_id": f"group_{label}_{collection}_{index}"})
    return pd.DataFrame(rows)


class M6OptimizationTests(unittest.TestCase):
    def test_training_manifest_reader_stops_at_holdout_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.csv"
            fields = ["filename", "primary_label", "collection", "group_id", "partition"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i in range(m6.EXPECTED_TRAIN_ROWS):
                    label = config.WORKING_LABELS[i % len(config.WORKING_LABELS)]
                    collection = "XC" if (i // len(config.WORKING_LABELS)) % 2 == 0 else "iNat"
                    writer.writerow({"filename": f"train_{i}.ogg", "primary_label": label,
                                     "collection": collection, "group_id": f"g{i}", "partition": "train"})
                # These values are deliberately invalid: the train reader must stop at the boundary.
                writer.writerow({"filename": "holdout_sentinel", "primary_label": "NOT_A_CLASS",
                                 "collection": "INVALID", "group_id": "leak", "partition": "validation"})
            frame = m6.read_training_metadata(path)
            self.assertEqual(len(frame), m6.EXPECTED_TRAIN_ROWS)
            self.assertNotIn("NOT_A_CLASS", set(frame.primary_label))

    def test_group_splits_are_deterministic_and_leakage_safe(self):
        metadata = small_metadata()
        first = m6.make_group_splits(metadata, n_splits=5, seed=42)
        second = m6.make_group_splits(metadata, n_splits=5, seed=42)
        np.testing.assert_array_equal(first[0][0], second[0][0])
        np.testing.assert_array_equal(first[0][1], second[0][1])
        groups = metadata.group_id.to_numpy()
        for train, valid in first:
            self.assertFalse(set(groups[train]) & set(groups[valid]))

    def test_learned_scaler_uses_fold_training_rows_only(self):
        metadata = small_metadata()
        splits = m6.make_group_splits(metadata)
        train, valid = splits[0]
        x = np.zeros((len(metadata), m6.EMBEDDING_DIM), dtype=np.float32)
        x[train] = 3.0
        x[valid] = 90.0
        params = {"C": 1.0, "class_weight": "balanced", "max_iter": 50, "random_state": 42}
        estimator = m6._make_estimator("logistic_regression", params, scale=True)
        estimator.fit(x[train], metadata.primary_label.to_numpy()[train])
        np.testing.assert_allclose(estimator.named_steps["scaler"].mean_, 3.0)

    def test_metric_calculation_matches_expected_macro_f1(self):
        y = np.array(config.WORKING_LABELS)
        pred = np.array(config.WORKING_LABELS)
        pred[-1] = pred[-2]

        class FixedPredictor:
            def predict(self, _x):
                return pred

        metrics, _ = m6._evaluate_fold(FixedPredictor(), np.zeros((4, 1)), y)
        self.assertAlmostEqual(metrics["macro_f1"], f1_score(y, pred, labels=list(config.WORKING_LABELS), average="macro"))

    def test_experiment_writes_consistent_model_and_summary_without_holdout(self):
        metadata = m6.read_training_metadata()
        rng = np.random.default_rng(42)
        x = rng.normal(size=(m6.EXPECTED_TRAIN_ROWS, m6.EMBEDDING_DIM)).astype(np.float32)
        one_candidate = [("logistic_regression",
                          {"C": 0.1, "class_weight": "balanced", "max_iter": 100,
                           "random_state": 42}, True)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(m6, "_candidate_specs", return_value=one_candidate):
                result = m6.run_optimization(x, metadata, root / "reports", root / "models",
                                             progress=lambda *_: None)
            model_path = result["model_path"]
            self.assertTrue(model_path.is_file())
            summary = result["summary"]
            self.assertFalse(summary["holdout_used"])
            self.assertEqual(summary["training_recordings"], 1748)
            self.assertEqual(summary["class_order"], list(config.WORKING_LABELS))
            self.assertEqual(len(result["folds"]), 5)
            metadata_payload = (root / "models/model_metadata.json").read_text(encoding="utf-8")
            self.assertIn(summary["final_model_sha256"], metadata_payload)
            self.assertIn('"holdout_used": false', metadata_payload)
            self.assertTrue((root / "reports/m6_optimization_report.md").is_file())
            self.assertTrue((root / "reports/selected_oof_confusion_matrix.csv").is_file())

    def test_existing_output_directories_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            models = root / "models"
            reports.mkdir()
            with self.assertRaises(FileExistsError):
                m6.run_optimization(np.zeros((1, m6.EMBEDDING_DIM)), pd.DataFrame(), reports, models)


if __name__ == "__main__":
    unittest.main()
