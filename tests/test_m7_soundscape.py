"""Tests for M7 annotation parsing and multi-label utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biosonic_ml.m7_soundscape import (
    SoundscapeAnnotationError,
    audit_local_soundscape_data,
    assert_group_disjoint,
    build_multihot_targets,
    format_window_predictions,
    parse_label_cell,
    sigmoid_probabilities,
    timestamp_seconds,
    validate_soundscape_annotations,
)
from download_m7_soundscapes import _safe_filename, build_manifest


def toy_taxonomy() -> pd.DataFrame:
    return pd.DataFrame({
        "primary_label": ["bird_a", "bird_b", "frog_a"],
        "class_name": ["Aves", "Aves", "Amphibia"],
        "common_name": ["Bird A", "Bird B", "Frog A"],
    })


def toy_annotations() -> pd.DataFrame:
    return pd.DataFrame([
        {"filename": "site1.ogg", "start": "00:00:00", "end": "00:00:05", "primary_label": "bird_a;frog_a"},
        {"filename": "site1.ogg", "start": "00:00:10", "end": "00:00:15", "primary_label": "bird_b"},
    ])


class M7SoundscapeTests(unittest.TestCase):
    def test_acquisition_whitelist_matches_66_inventory_files(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest), 66)
        self.assertEqual(manifest.filename.nunique(), 66)
        self.assertEqual(int(manifest.expected_size_bytes.sum()), 33_785_154)
        self.assertTrue(manifest.source_path.str.startswith("train_soundscapes/").all())

    def test_acquisition_filename_rejects_paths_and_non_ogg(self):
        self.assertTrue(_safe_filename("BC2026_Train_0001_S01_20250101_000000.ogg"))
        for filename in ("../escape.ogg", "folder/file.ogg", "C:relative.ogg", "audio.wav", ""):
            self.assertFalse(_safe_filename(filename), filename)

    def test_timestamp_and_semicolon_label_parsing(self):
        self.assertEqual(timestamp_seconds("00:01:05"), 65)
        self.assertEqual(parse_label_cell("bird_a; frog_a"), ("bird_a", "frog_a"))
        with self.assertRaises(SoundscapeAnnotationError):
            timestamp_seconds("00:60:00")
        with self.assertRaises(SoundscapeAnnotationError):
            parse_label_cell("bird_a;;frog_a")

    def test_exact_duplicates_are_removed_but_conflicts_fail(self):
        source = pd.concat([toy_annotations().iloc[[0]], toy_annotations().iloc[[0]]], ignore_index=True)
        validated = validate_soundscape_annotations(source, toy_taxonomy())
        self.assertEqual(len(validated), 1)
        conflict = pd.concat([source.iloc[[0]], pd.DataFrame([{
            "filename": "site1.ogg", "start": "00:00:00", "end": "00:00:05", "primary_label": "bird_b"
        }])], ignore_index=True)
        with self.assertRaises(SoundscapeAnnotationError):
            validate_soundscape_annotations(conflict, toy_taxonomy())

    def test_unknown_label_and_partial_interval_are_rejected(self):
        unknown = toy_annotations().iloc[[0]].copy()
        unknown.loc[:, "primary_label"] = "not_in_taxonomy"
        with self.assertRaisesRegex(SoundscapeAnnotationError, "absent from taxonomy"):
            validate_soundscape_annotations(unknown, toy_taxonomy())
        partial = toy_annotations().iloc[[0]].copy()
        partial.loc[:, "end"] = "00:00:04"
        with self.assertRaisesRegex(SoundscapeAnnotationError, "duration 5 seconds"):
            validate_soundscape_annotations(partial, toy_taxonomy())

    def test_targets_are_multihot_and_missing_windows_are_not_synthesized(self):
        validated = validate_soundscape_annotations(toy_annotations(), toy_taxonomy())
        windows, targets, labels = build_multihot_targets(validated, toy_taxonomy())
        self.assertEqual(labels, ("bird_a", "bird_b"))
        self.assertEqual(windows.start_seconds.tolist(), [0, 10])
        np.testing.assert_array_equal(targets, np.asarray([[1, 0], [0, 1]], dtype=np.uint8))
        self.assertEqual(len(windows), 2)  # 00:00:05-00:00:10 was not annotated.

    def test_annotated_nonbird_window_is_zero_for_bird_target(self):
        source = pd.DataFrame([{
            "filename": "site1.ogg", "start": "00:00:00", "end": "00:00:05", "primary_label": "frog_a"
        }])
        validated = validate_soundscape_annotations(source, toy_taxonomy())
        _, targets, _ = build_multihot_targets(validated, toy_taxonomy())
        self.assertEqual(targets.shape, (1, 0))

    def test_group_disjointness_detects_recording_leakage(self):
        train = pd.DataFrame({"group_id": ["site_a", "site_b"]})
        valid = pd.DataFrame({"group_id": ["site_c"]})
        assert_group_disjoint(train, valid)
        with self.assertRaisesRegex(SoundscapeAnnotationError, "Group leakage"):
            assert_group_disjoint(train, pd.DataFrame({"group_id": ["site_b"]}))

    def test_sigmoid_outputs_are_independent_probabilities(self):
        actual = sigmoid_probabilities(np.asarray([[4.0, 4.0, -4.0]]))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
        self.assertAlmostEqual(float(actual[0, 0]), float(actual[0, 1]))
        self.assertGreater(float(actual[0, 0] + actual[0, 1]), 1.0)

    def test_timestamped_prediction_format_supports_multiple_and_none(self):
        windows = pd.DataFrame([
            {"filename": "validation.ogg", "start_seconds": 0, "end_seconds": 5},
            {"filename": "validation.ogg", "start_seconds": 5, "end_seconds": 10},
        ])
        result = format_window_predictions(
            windows, np.asarray([[0.8, 0.7], [0.1, 0.2]]), ("bird_a", "bird_b"), thresholds=0.5
        )
        self.assertEqual(result[0]["species"], ["bird_a", "bird_b"])
        self.assertFalse(result[0]["no_supported_species_above_threshold"])
        self.assertEqual(result[1]["species"], [])
        self.assertTrue(result[1]["no_supported_species_above_threshold"])
        self.assertEqual(result[0]["start_seconds"], 0)
        self.assertEqual(result[0]["end_seconds"], 5)

    def test_local_audit_counts_inventory_without_claiming_audio_is_present(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "metadata"
            metadata.mkdir()
            filename = "BC2026_Train_0001_S01_20250101_000000.ogg"
            ann = pd.DataFrame([
                {"filename": filename, "start": "00:00:00", "end": "00:00:05", "primary_label": "bird_a;frog_a"},
                {"filename": filename, "start": "00:00:00", "end": "00:00:05", "primary_label": "bird_a;frog_a"},
            ])
            inv = pd.DataFrame([
                {"FileName": f"train_soundscapes/{filename}", "SizeBytes": 123, "Created": "x"},
                {"FileName": "train_audio/bird_a/example.ogg", "SizeBytes": 20, "Created": "x"},
            ])
            ann.to_csv(metadata / "labels.csv", index=False)
            inv.to_csv(metadata / "inventory.csv", index=False)
            toy_taxonomy().to_csv(metadata / "taxonomy.csv", index=False)
            pd.DataFrame({"filename": ["bird_a/example.ogg"], "primary_label": ["bird_a"]}).to_csv(
                metadata / "train.csv", index=False
            )
            audit = audit_local_soundscape_data(
                annotation_path=metadata / "labels.csv",
                taxonomy_path=metadata / "taxonomy.csv",
                inventory_path=metadata / "inventory.csv",
                train_metadata_path=metadata / "train.csv",
                data_root=root / "data",
            )
            self.assertEqual(audit["annotation_exact_duplicates_removed"], 1)
            self.assertEqual(audit["inventory_train_soundscape_files"], 1)
            self.assertEqual(audit["annotated_audio_files_present_locally"], 0)
            self.assertEqual(audit["annotated_soundscapes"], 1)
            self.assertEqual(audit["bird_species_count"], 1)


if __name__ == "__main__":
    unittest.main()
