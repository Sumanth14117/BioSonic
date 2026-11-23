import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosonic_ml import config
from biosonic_ml.dataset import DatasetError, load_working_dataset
from biosonic_ml.features import (extract_feature_table, feature_configuration, recording_features,
                                  segment_feature)
from biosonic_ml.splitting import make_holdout
from biosonic_ml.training import _evaluate, _estimators, make_group_cv_splits


def make_fixture(root: Path, *, duplicate=False, sr=32000, channels=1):
    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    train, manifest, inventory = [], [], []
    first_audio = None
    n = 0
    for label in config.WORKING_LABELS:
        for collection in ("XC", "iNat"):
            for author_i in range(6):
                author = f"author-{label}-{collection}-{author_i}"
                # A single shared author across labels verifies global grouping.
                if author_i == 0 and collection == "XC":
                    author = "shared-author"
                name = f"{label}/XC{100000+n}.ogg"
                path = audio / name
                path.parent.mkdir(parents=True, exist_ok=True)
                seed = 100 if duplicate and n == 1 else 100 + n
                samples = np.random.default_rng(seed).normal(0, 0.1, max(1024, sr // 10)).astype(np.float32)
                sf.write(path, np.tile(samples[:, None], (1, channels)), sr, format="OGG", subtype="VORBIS")
                if first_audio is None:
                    first_audio = path
                elif duplicate and n == 1:
                    shutil.copyfile(first_audio, path)
                size = path.stat().st_size
                train.append({"filename": name, "primary_label": label, "author": author,
                              "collection": collection, "class_name": "Aves"})
                manifest.append({"filename": name, "primary_label": label, "verification_status": "verified",
                                 "expected_size_bytes": size})
                inventory.append({"FileName": "train_audio/" + name, "SizeBytes": size})
                n += 1
    # Literal Unknowns are one global group and need to remain in training.
    for label in config.WORKING_LABELS:
        name = f"{label}/XC{100000+n}.ogg"
        path = audio / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(3200, dtype=np.float32), 32000, format="OGG", subtype="VORBIS")
        size = path.stat().st_size
        train.append({"filename": name, "primary_label": label, "author": "Unknown", "collection": "XC", "class_name": "Aves"})
        manifest.append({"filename": name, "primary_label": label, "verification_status": "verified", "expected_size_bytes": size})
        inventory.append({"FileName": "train_audio/" + name, "SizeBytes": size})
        n += 1
    pd.DataFrame(train).to_csv(root / "train.csv", index=False)
    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(inventory).to_csv(root / "inventory.csv", index=False)
    pd.DataFrame({"primary_label": config.WORKING_LABELS, "class_name": ["Aves"] * len(config.WORKING_LABELS)}).to_csv(root / "taxonomy.csv", index=False)
    return {"manifest_path": root / "manifest.csv", "train_path": root / "train.csv",
            "inventory_path": root / "inventory.csv", "taxonomy_path": root / "taxonomy.csv", "audio_root": audio,
            "count": n}


def fixture_frame(root, **kwargs):
    args = make_fixture(root, **kwargs)
    frame = load_working_dataset(**{k: v for k, v in args.items() if k != "count"}, expected_count=args["count"])
    return args, frame


def test_interim_scope_contains_eight_present_labels_and_exact_identifier():
    assert len(config.WORKING_LABELS) == 8
    assert "smbani" not in config.WORKING_LABELS
    assert "smbani" in config.ALL_APPROVED_LABELS


def test_m2_sample_names_are_outside_approved_working_scope():
    assert len(config.SAMPLE_FILES) == 6
    assert not set(config.SAMPLE_FILES) & set(config.EXCLUDED_FILES)


def test_load_dataset_validates_metadata_inventory_and_audio(tmp_path):
    args, frame = fixture_frame(tmp_path)
    assert len(frame) == args["count"] == 104
    assert set(frame.primary_label) == set(config.WORKING_LABELS)
    assert (frame.sample_rate == 32000).all() and (frame.channels == 1).all()


def test_loader_rejects_file_size_mismatch(tmp_path):
    args = make_fixture(tmp_path)
    manifest = pd.read_csv(args["manifest_path"])
    manifest.loc[0, "expected_size_bytes"] += 1
    manifest.to_csv(args["manifest_path"], index=False)
    with pytest.raises(DatasetError, match="Inventory missing or size disagrees|File-size mismatch"):
        load_working_dataset(**{k: v for k, v in args.items() if k != "count"}, expected_count=args["count"])


def test_loader_rejects_missing_audio(tmp_path):
    args = make_fixture(tmp_path)
    first = pd.read_csv(args["manifest_path"]).iloc[0].filename
    (args["audio_root"] / first).unlink()
    with pytest.raises(DatasetError, match="Verified file is absent"):
        load_working_dataset(**{k: v for k, v in args.items() if k != "count"}, expected_count=args["count"])


def test_loader_rejects_metadata_label_mismatch(tmp_path):
    args = make_fixture(tmp_path)
    train = pd.read_csv(args["train_path"])
    train.loc[0, "primary_label"] = config.WORKING_LABELS[1]
    train.to_csv(args["train_path"], index=False)
    with pytest.raises(DatasetError, match="Metadata label/class mismatch"):
        load_working_dataset(**{k: v for k, v in args.items() if k != "count"}, expected_count=args["count"])


def test_loader_rejects_audio_sample_rate_and_channel_mismatch(tmp_path):
    args = make_fixture(tmp_path)
    first = pd.read_csv(args["manifest_path"]).iloc[0].filename
    path = args["audio_root"] / first
    sf.write(path, np.zeros((3200, 2), dtype=np.float32), 44100, format="OGG", subtype="VORBIS")
    manifest = pd.read_csv(args["manifest_path"])
    manifest.loc[0, "expected_size_bytes"] = path.stat().st_size
    manifest.to_csv(args["manifest_path"], index=False)
    inv = pd.read_csv(args["inventory_path"])
    inv.loc[0, "SizeBytes"] = path.stat().st_size
    inv.to_csv(args["inventory_path"], index=False)
    with pytest.raises(DatasetError, match="Audio validation failed|Expected mono"):
        load_working_dataset(**{k: v for k, v in args.items() if k != "count"}, expected_count=args["count"])


def test_unknown_authors_are_one_group(tmp_path):
    _, frame = fixture_frame(tmp_path)
    unknown = frame[frame.author_group_key.eq("__unknown__")]
    assert len(unknown) == 8
    assert unknown.group_id.nunique() == 1


def test_author_identity_is_grouped_across_species(tmp_path):
    _, frame = fixture_frame(tmp_path)
    shared = frame[frame.author.eq("shared-author")]
    assert shared.primary_label.nunique() == 8
    assert shared.group_id.nunique() == 1


def test_duplicate_audio_hashes_are_joined_as_one_group(tmp_path):
    _, frame = fixture_frame(tmp_path, duplicate=True)
    assert frame.sha256.duplicated().any()
    assert frame.groupby("sha256").group_id.nunique().max() == 1


def test_split_is_deterministic_and_approximately_eighty_twenty(tmp_path):
    _, frame = fixture_frame(tmp_path)
    outputs = []
    for i in (1, 2):
        outputs.append(make_holdout(frame, tmp_path / f"split{i}.csv", tmp_path / f"sum{i}.json", tmp_path / f"strat{i}.md")[0])
    assert outputs[0][["filename", "partition"]].equals(outputs[1][["filename", "partition"]])
    ratio = outputs[0].partition.eq("validation").mean()
    assert 0.15 <= ratio <= 0.25


def test_split_has_no_author_or_duplicate_group_overlap(tmp_path):
    _, frame = fixture_frame(tmp_path)
    split, _ = make_holdout(frame, tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "s.md")
    assert not set(split.loc[split.partition.eq("train"), "group_id"]) & set(split.loc[split.partition.eq("validation"), "group_id"])


def test_unknown_author_records_remain_in_training(tmp_path):
    _, frame = fixture_frame(tmp_path)
    split, summary = make_holdout(frame, tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "s.md")
    assert split.loc[split.unknown_author, "partition"].eq("train").all()
    assert summary["unknown_author_validation"] == 0


def test_each_class_and_collection_stratum_in_both_partitions(tmp_path):
    _, frame = fixture_frame(tmp_path)
    split, summary = make_holdout(frame, tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "s.md")
    assert split.groupby("partition").primary_label.nunique().eq(8).all()
    assert split.groupby(["partition", "primary_label", "collection"]).size().min() >= 1
    assert summary["all_label_collection_strata_present_in_both"]


def test_split_manifest_preserves_recording_source_hash_and_size(tmp_path):
    _, frame = fixture_frame(tmp_path)
    split, _ = make_holdout(frame, tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "s.md")
    assert set(("filename", "sha256", "actual_size_bytes", "partition", "group_id")).issubset(split.columns)


def test_feature_configuration_is_explicit_and_power_mel():
    cfg = feature_configuration()
    assert cfg["sample_rate"] == 32000 and cfg["mel_bands"] == 128
    assert cfg["spectrogram"] == "power" and cfg["mel_scale"] == "Slaney (htk=False)"
    assert cfg["mel_norm"] == "Slaney" and cfg["n_fft"] == cfg["win_length"] == 1024


def test_complete_segment_feature_shape_and_finiteness():
    x = np.random.default_rng(5).normal(0, 0.05, config.SEGMENT_SAMPLES).astype(np.float32)
    feature, frames = segment_feature(x, len(x))
    assert feature.shape == (len(config.FEATURE_NAMES),)
    assert frames == 622
    assert np.isfinite(feature).all()


def test_short_recording_is_zero_padded_and_tail_is_reported(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.random.default_rng(3).normal(0, 0.02, 8000).astype(np.float32), 32000)
    vector, info = recording_features(path)
    assert vector.shape == (2 * len(config.FEATURE_NAMES),)
    assert info["segment_count"] == 1 and info["padded_samples"] == 152000
    assert info["partial_or_short_segments"] == 1 and info["source_sample_coverage"] == 1


def test_feature_rejects_nonfinite_waveform():
    x = np.zeros(config.SEGMENT_SAMPLES, dtype=np.float32)
    x[10] = np.nan
    with pytest.raises(ValueError, match="Invalid segment samples"):
        segment_feature(x, len(x))


def test_feature_rejects_wrong_sample_rate():
    with pytest.raises(ValueError, match="Expected 32000 Hz"):
        segment_feature(np.zeros(config.SEGMENT_SAMPLES), config.SEGMENT_SAMPLES, 44100)


def test_resumable_feature_shards_are_reused(tmp_path, monkeypatch):
    _, frame = fixture_frame(tmp_path / "data")
    split = frame.iloc[:1].copy()
    split["partition"] = "train"
    import biosonic_ml.features as feature_mod
    calls = []
    monkeypatch.setattr(feature_mod, "recording_features", lambda path: (calls.append(path) or np.ones(2 * len(config.FEATURE_NAMES), dtype=np.float32), {"segment_count": 1}))
    X1, m1 = extract_feature_table(split, tmp_path / "out")
    X2, m2 = extract_feature_table(split, tmp_path / "out")
    assert len(calls) == 1 and np.array_equal(X1, X2) and m1.filename.tolist() == m2.filename.tolist()


def test_incompatible_feature_config_stops_without_overwrite(tmp_path, monkeypatch):
    _, frame = fixture_frame(tmp_path / "data")
    split = frame.iloc[:1].copy()
    split["partition"] = "train"
    import biosonic_ml.features as feature_mod
    monkeypatch.setattr(feature_mod, "recording_features", lambda path: (np.ones(2 * len(config.FEATURE_NAMES), dtype=np.float32), {"segment_count": 1}))
    extract_feature_table(split, tmp_path / "out")
    cfg = json.loads((tmp_path / "out/feature_config.json").read_text())
    cfg["config_hash"] = "wrong"
    (tmp_path / "out/feature_config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="differs"):
        extract_feature_table(split, tmp_path / "out")


def test_training_cv_keeps_groups_separate_and_classes_present(tmp_path):
    _, frame = fixture_frame(tmp_path)
    split, _ = make_holdout(frame, tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "s.md")
    train = split[split.partition.eq("train")].reset_index(drop=True)
    folds = make_group_cv_splits(train)
    assert len(folds) == 5
    assert all(not set(train.iloc[a].group_id) & set(train.iloc[b].group_id) for a, b in folds)
    assert all(set(train.iloc[a].primary_label) == set(config.WORKING_LABELS) for a, b in folds)


def test_scaler_is_inside_lr_and_svm_pipelines_only():
    estimators = _estimators(42)
    assert isinstance(estimators["logistic_regression"][0].named_steps["scaler"], StandardScaler)
    assert isinstance(estimators["svm"][0].named_steps["scaler"], StandardScaler)
    assert "scaler" not in estimators["random_forest"][0].named_steps


def test_svm_roc_auc_probabilities_are_not_enabled():
    estimator = _estimators(42)["svm"][0].named_steps["model"]
    # sklearn 1.9 uses a 'deprecated' sentinel for its default-false behavior;
    # explicitly setting False emits a FutureWarning, so the parameter is omitted.
    assert estimator.probability == "deprecated"


def test_roc_auc_probability_columns_follow_sorted_label_order():
    labels = list(config.WORKING_LABELS)
    sorted_labels = sorted(labels)
    y = np.asarray(labels, dtype=str)

    class ProbabilityModel:
        classes_ = np.asarray(sorted_labels, dtype=str)

        def predict(self, X):
            return y

        def predict_proba(self, X):
            result = np.zeros((len(y), len(sorted_labels)))
            for row, label in enumerate(y):
                result[row, sorted_labels.index(label)] = 1.0
            return result

    metrics = _evaluate(ProbabilityModel(), np.zeros((len(y), 1)), y, labels)
    assert metrics["roc_auc_ovr_macro"] == pytest.approx(1.0)


def test_synthetic_model_trains_and_predicts_all_classes():
    rng = np.random.default_rng(18)
    y = np.repeat(np.asarray(config.WORKING_LABELS), 8)
    centers = np.repeat(np.eye(len(config.WORKING_LABELS)), 8, axis=0)
    X = np.column_stack([centers, rng.normal(0, 0.02, (len(y), 3))])
    model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions) == set(config.WORKING_LABELS)
    assert (predictions == y).mean() > 0.95


def test_metric_calculation_matches_expected_accuracy_and_macro_f1():
    labels = list(config.WORKING_LABELS)
    y = np.asarray(labels)
    predicted = y.copy()
    predicted[0] = predicted[1]

    class FixedModel:
        def predict(self, X):
            return predicted

    metrics = _evaluate(FixedModel(), np.zeros((len(y), 1)), y, labels)
    assert metrics["accuracy"] == pytest.approx(7 / 8)
    assert metrics["f1_macro"] < 1.0


def test_group_cv_does_not_accept_incomplete_training_labels(tmp_path):
    _, frame = fixture_frame(tmp_path)
    with pytest.raises(ValueError, match="all eight labels"):
        make_group_cv_splits(frame[frame.primary_label.ne(config.WORKING_LABELS[-1])])
