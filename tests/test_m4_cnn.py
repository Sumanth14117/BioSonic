import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosonic_ml import config as m3config
from biosonic_cnn import config
from biosonic_cnn.data import make_internal_partition, make_segment_manifest
from biosonic_cnn.dataset import MelWindowDataset
from biosonic_cnn.mels import compute_mel_window, fit_train_normalizer, make_segment_index, mel_config
from biosonic_cnn.model import BioSonicCNN, architecture_config, parameter_count
from biosonic_cnn.train import (_atomic_torch_save, _load_model_state, aggregate_recording_probabilities,
                                calculate_recording_metrics, seed_everything)
from biosonic_cnn.pipeline import _plot_holdout_diagnostics


def synthetic_split():
    rows = []
    file_id = 0
    for label in config.INDEX_TO_LABEL:
        for collection in ("XC", "iNat"):
            for author_no in range(12):
                author = f"author-{author_no}-{collection}"
                # Share a known author across species to test global grouping.
                if author_no == 0:
                    author = "shared-across-labels"
                rows.append({"filename": f"{label}/XC{file_id}.ogg", "primary_label": label,
                             "collection": collection, "group_id": f"g-{author}",
                             "author_group_id": f"a-{author}", "sha256": f"hash-{file_id}",
                             "expected_size_bytes": 1, "actual_size_bytes": 1, "sample_rate": 32000,
                             "channels": 1, "sample_count": 160000, "duration_seconds": 5.0,
                             "partition": "train", "unknown_author": False})
                file_id += 1
    for label in config.INDEX_TO_LABEL:
        rows.append({"filename": f"{label}/unknown.ogg", "primary_label": label,
                     "collection": "XC", "group_id": "unknown-global", "author_group_id": "unknown-global",
                     "sha256": f"hash-u-{label}", "expected_size_bytes": 1, "actual_size_bytes": 1,
                     "sample_rate": 32000, "channels": 1, "sample_count": 160000, "duration_seconds": 5.0,
                     "partition": "train", "unknown_author": True})
    return pd.DataFrame(rows)


def test_correct_eight_class_mapping():
    assert config.LABEL_TO_INDEX == {label: i for i, label in enumerate(config.INDEX_TO_LABEL)}
    assert len(config.LABEL_TO_INDEX) == 8
    assert "smbani" not in config.LABEL_TO_INDEX


def test_holdout_diagnostic_plots_use_saved_reports_only(tmp_path, monkeypatch):
    m4_dir, m3_dir = tmp_path / "m4", tmp_path / "m3"
    m4_dir.mkdir()
    m3_dir.mkdir()
    monkeypatch.setattr(config, "M4_REPORT_DIR", m4_dir)
    monkeypatch.setattr(config, "M3_REPORT_DIR", m3_dir)
    labels = config.INDEX_TO_LABEL
    matrix = pd.DataFrame(np.eye(len(labels), dtype=int), index=labels, columns=labels)
    matrix.to_csv(m4_dir / "holdout_confusion_matrix.csv")
    for model in ("logistic_regression", "random_forest", "svm"):
        pd.DataFrame({"class": labels, "f1-score": np.linspace(.4, .8, len(labels))}).to_csv(
            m3_dir / f"{model}_classification_report.csv", index=False)
    pd.DataFrame({"class": labels, "f1-score": np.linspace(.5, .9, len(labels))}).to_csv(
        m4_dir / "holdout_classification_report.csv", index=False)

    _plot_holdout_diagnostics()

    assert (m4_dir / "holdout_confusion_matrix.png").is_file()
    assert (m4_dir / "per_class_holdout_f1_comparison.png").is_file()


def test_internal_split_is_deterministic_and_within_requested_range(tmp_path):
    frame = synthetic_split()
    out1, summary1 = make_internal_partition(frame, tmp_path / "split1.csv", tmp_path / "summary1.json",
                                             expected_training_count=len(frame))
    out2, summary2 = make_internal_partition(frame, tmp_path / "split2.csv", tmp_path / "summary2.json",
                                             expected_training_count=len(frame))
    assert out1[["filename", "m4_partition"]].equals(out2[["filename", "m4_partition"]])
    fraction = summary1["internal_validation_fraction_of_m3_train"]
    assert 0.15 <= fraction <= 0.20
    assert summary1["selected_fold"] == summary2["selected_fold"]


def test_no_recording_author_or_duplicate_group_leakage(tmp_path):
    frame = synthetic_split()
    split, _ = make_internal_partition(frame, tmp_path / "split.csv", tmp_path / "summary.json",
                                       expected_training_count=len(frame))
    inner = split[split.m4_partition.isin(["internal_train", "internal_validation"])]
    tr, va = inner[inner.m4_partition.eq("internal_train")], inner[inner.m4_partition.eq("internal_validation")]
    assert not set(tr.filename) & set(va.filename)
    assert not set(tr.group_id) & set(va.group_id)
    assert not set(tr.author_group_id) & set(va.author_group_id)


def test_unknown_authors_stay_in_internal_training(tmp_path):
    split, summary = make_internal_partition(synthetic_split(), tmp_path / "split.csv", tmp_path / "summary.json",
                                             expected_training_count=len(synthetic_split()))
    assert split.loc[split.unknown_author, "m4_partition"].eq("internal_train").all()
    assert summary["unknown_author_internal_validation"] == 0


def test_all_classes_and_collection_strata_in_internal_partitions(tmp_path):
    split, _ = make_internal_partition(synthetic_split(), tmp_path / "split.csv", tmp_path / "summary.json",
                                       expected_training_count=len(synthetic_split()))
    internal = split[split.m4_partition.isin(["internal_train", "internal_validation"])]
    for partition in ("internal_train", "internal_validation"):
        part = internal[internal.m4_partition.eq(partition)]
        assert set(part.primary_label) == set(config.INDEX_TO_LABEL)
        assert set(part.collection) == {"XC", "iNat"}
        assert part.groupby(["primary_label", "collection"]).ngroups == 16


def test_existing_m3_holdout_is_retained_exactly(tmp_path):
    frame = synthetic_split()
    holdout = frame.iloc[:24].copy()
    holdout["partition"] = "validation"
    holdout["filename"] = [f"prior-holdout/{i}.ogg" for i in range(len(holdout))]
    holdout["group_id"] = [f"held-group-{i}" for i in range(len(holdout))]
    holdout["author_group_id"] = [f"held-author-{i}" for i in range(len(holdout))]
    frame = pd.concat([frame, holdout], ignore_index=True)
    split, _ = make_internal_partition(frame, tmp_path / "split.csv", tmp_path / "summary.json",
                                       expected_training_count=len(frame) - len(holdout))
    assert split.m4_partition.eq("holdout").sum() == len(holdout)
    assert (split.loc[split.m4_partition.eq("holdout"), "partition"] == "validation").all()


def test_segment_mapping_inherits_recording_partition(tmp_path):
    recordings = synthetic_split().head(16).copy()
    recordings["m4_partition"] = ["internal_train"] * 8 + ["internal_validation"] * 8
    segments = recordings[["filename", "primary_label", "collection", "group_id", "sha256"]].copy()
    segments["segment_index"] = 0
    result = make_segment_manifest(recordings, segments, tmp_path / "segments.csv")
    assert result.m4_partition.tolist() == recordings.m4_partition.tolist()


def test_mel_spectrogram_shape_for_complete_window():
    waveform = np.random.default_rng(2).normal(0, 0.05, 160000).astype(np.float32)
    mel, valid_frames = compute_mel_window(waveform, len(waveform))
    assert mel.shape == (128, 622)
    assert valid_frames == 622
    assert np.isfinite(mel).all()


def test_short_audio_and_final_tail_are_zero_padded_and_reported():
    waveform = np.zeros(160000, dtype=np.float32)
    mel, valid_frames = compute_mel_window(waveform, 4000)
    assert mel.shape == (128, 622)
    assert valid_frames == 12
    _, tiny_tail_frames = compute_mel_window(waveform, 17)
    assert tiny_tail_frames == 1


def test_documented_mel_config_matches_m3_exploratory_settings():
    settings = mel_config()
    assert settings["sample_rate"] == 32000
    assert settings["n_fft"] == settings["win_length"] == 1024
    assert settings["hop_length"] == 256 and settings["mel_bands"] == 128
    assert settings["mel_scale"] == "Slaney (htk=False)" and settings["mel_filter_normalization"] == "Slaney"


def test_dataset_shape_label_and_metadata(tmp_path):
    cache = tmp_path / "cache"
    filename = "compot1/XC1.ogg"
    key = __import__("hashlib").sha256(filename.encode()).hexdigest()[:24]
    (cache / key).mkdir(parents=True)
    np.save(cache / key / "mels.npy", np.ones((1, 128, 622), dtype=np.float16))
    index = pd.DataFrame([{"filename": filename, "primary_label": "compot1", "m4_partition": "internal_train",
                           "segment_index": 0, "valid_frames": 622, "sha256": "abc", "author_group_id": "a",
                           "collection": "XC", "source_sample_count": 160000, "padded_sample_count": 0}])
    normalizer = {"fit_partition": "internal_train", "mean": [0] * 128, "std": [1] * 128}
    item = MelWindowDataset(index, normalizer, cache)[0]
    assert item["x"].shape == (1, 128, 622)
    assert item["x"].dtype == torch.float32 and item["y"].item() == config.LABEL_TO_INDEX["compot1"]


def test_padding_is_neutral_after_training_only_normalization(tmp_path):
    cache = tmp_path / "cache"
    filename = "compot1/XC2.ogg"
    key = __import__("hashlib").sha256(filename.encode()).hexdigest()[:24]
    (cache / key).mkdir(parents=True)
    np.save(cache / key / "mels.npy", np.ones((1, 128, 622), dtype=np.float16))
    index = pd.DataFrame([{"filename": filename, "primary_label": "compot1", "m4_partition": "internal_train",
                           "segment_index": 0, "valid_frames": 100, "sha256": "a", "author_group_id": "g",
                           "collection": "XC", "source_sample_count": 26000, "padded_sample_count": 134000}])
    normalizer = {"fit_partition": "internal_train", "mean": [0] * 128, "std": [1] * 128}
    x = MelWindowDataset(index, normalizer, cache)[0]["x"].numpy()
    assert np.all(x[:, :, 100:] == 0)


def test_normalization_statistics_use_internal_training_only(tmp_path):
    cache = tmp_path / "cache"
    rows = []
    for filename, partition, value in (("compot1/train.ogg", "internal_train", 2.0),
                                       ("rufhor2/hold.ogg", "holdout", 999.0),
                                       ("saytan1/val.ogg", "internal_validation", 500.0)):
        key = __import__("hashlib").sha256(filename.encode()).hexdigest()[:24]
        folder = cache / key
        folder.mkdir(parents=True)
        np.save(folder / "mels.npy", np.full((1, 128, 622), value, dtype=np.float16))
        (folder / "metadata.json").write_text(json.dumps({"partition": partition, "valid_frames": [4]}))
        rows.append({"filename": filename, "m4_partition": partition, "segment_index": 0, "valid_frames": 4})
    normalizer = fit_train_normalizer(pd.DataFrame(rows), cache, tmp_path / "norm.json")
    assert normalizer["fit_partition"] == "internal_train"
    assert normalizer["fit_recordings"] == normalizer["fit_segments"] == 1
    assert normalizer["mean"][0] == pytest.approx(2.0)


def test_mel_segment_index_attaches_valid_frame_counts(tmp_path):
    cache = tmp_path / "cache"
    filename = "compot1/test.ogg"
    key = __import__("hashlib").sha256(filename.encode()).hexdigest()[:24]
    folder = cache / key
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"valid_frames": [622, 12]}))
    frame = pd.DataFrame([{"filename": filename, "segment_index": 0},
                          {"filename": filename, "segment_index": 1}])
    result = make_segment_index(frame, cache, tmp_path / "index.csv")
    assert result.valid_frames.tolist() == [622, 12]


def test_cnn_output_shape_and_parameter_count():
    model = BioSonicCNN()
    logits = model(torch.randn(2, 1, 128, 622))
    assert logits.shape == (2, 8)
    assert parameter_count(model) == architecture_config(model)["trainable_parameters"]


def test_loss_and_backpropagation_work():
    model = BioSonicCNN(channels=(4, 8, 12, 16))
    x, y = torch.randn(2, 1, 128, 622), torch.tensor([0, 3])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert next(model.parameters()).grad is not None


def test_checkpoint_saves_and_loads_model(tmp_path):
    model = BioSonicCNN(channels=(4, 8, 12, 16))
    path = tmp_path / "model.pt"
    _atomic_torch_save({"model_state": model.state_dict(), "epoch": 3}, path)
    loaded_state = torch.load(path, map_location="cpu", weights_only=False)
    clone = BioSonicCNN(channels=(4, 8, 12, 16))
    clone.load_state_dict(loaded_state["model_state"])
    assert loaded_state["epoch"] == 3
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))


def test_segment_probabilities_are_averaged_per_recording():
    probs = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    names, y, agg = aggregate_recording_probabilities(probs, ["a", "a", "b"], [0, 0, 1])
    assert names == ["a", "b"]
    assert y.tolist() == [0, 1]
    assert np.allclose(agg, [[0.5, 0.5], [0.1, 0.9]])


def test_recording_level_metrics_calculate_expected_values():
    y = np.arange(8)
    probs = np.eye(8) * 0.9 + (np.ones((8, 8)) - np.eye(8)) * (0.1 / 7)
    metrics = calculate_recording_metrics(y, probs)
    assert metrics["accuracy"] == 1.0 and metrics["f1_macro"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == pytest.approx(1.0)


def test_seed_reproducibility():
    seed_everything(42)
    first = torch.rand(4)
    seed_everything(42)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_holdout_partition_is_never_used_by_training_loader(tmp_path):
    cache = tmp_path / "cache"
    rows = []
    for i, (label, part) in enumerate((("compot1", "internal_train"), ("rufhor2", "holdout"))):
        name = f"{label}/{i}.ogg"
        key = __import__("hashlib").sha256(name.encode()).hexdigest()[:24]
        (cache / key).mkdir(parents=True)
        np.save(cache / key / "mels.npy", np.zeros((1, 128, 622), dtype=np.float16))
        rows.append({"filename": name, "primary_label": label, "m4_partition": part, "segment_index": 0,
                     "valid_frames": 622, "sha256": str(i), "author_group_id": str(i),
                     "collection": "XC", "source_sample_count": 160000, "padded_sample_count": 0})
    index = pd.DataFrame(rows)
    normalizer = {"fit_partition": "internal_train", "mean": [0] * 128, "std": [1] * 128}
    train = index[index.m4_partition.eq("internal_train")]
    loader = DataLoader(MelWindowDataset(train, normalizer, cache), batch_size=1)
    assert [x for batch in loader for x in batch["filename"]] == ["compot1/0.ogg"]


def test_mel_shard_creation_is_atomic_and_resumable(tmp_path, monkeypatch):
    import hashlib
    import biosonic_cnn.mels as mel_module
    audio_root = tmp_path / "audio"
    path = audio_root / "compot1" / "short.ogg"
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(4000, dtype=np.float32), 32000, format="OGG", subtype="VORBIS")
    monkeypatch.setattr(mel_module.config, "AUDIO_ROOT", audio_root)
    record = pd.DataFrame([{"filename": "compot1/short.ogg", "expected_size_bytes": path.stat().st_size,
                            "sha256": mel_module.sha256_file(path), "sample_count": 4000,
                            "m4_partition": "internal_train", "primary_label": "compot1",
                            "source_sample_count": 4000, "segment_index": 0}])
    cache = tmp_path / "mels"
    first = mel_module.extract_recording_mels(record, cache)
    second = mel_module.extract_recording_mels(record, cache)
    assert first == second
    assert first["shape"] == [1, 128, 622]
    assert len(list(cache.glob("*.tmp-*"))) == 0


def test_mel_shard_promotion_retries_windows_permission_error(tmp_path, monkeypatch):
    import biosonic_cnn.mels as mel_module
    source = tmp_path / "tmp-shard"
    destination = tmp_path / "final-shard"
    source.mkdir()
    np.save(source / "mels.npy", np.zeros((1, 128, 622), dtype=np.float16))
    metadata = {"filename": "compot1/test.ogg", "source_sha256": "digest", "config_hash": "config",
                "segments": 1, "valid_frames": [622]}
    (source / "metadata.json").write_text(json.dumps(metadata))
    calls, waits = [], []
    real_replace = mel_module.os.replace

    def flaky_replace(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError("transient Windows file lock")
        return real_replace(src, dst)

    monkeypatch.setattr(mel_module.os, "replace", flaky_replace)
    monkeypatch.setattr(mel_module.time, "sleep", waits.append)
    resumed = mel_module._promote_complete_shard(source, destination, "compot1/test.ogg", "digest", "config")
    assert resumed["filename"] == "compot1/test.ogg"
    assert len(calls) == 2 and waits == [0.1]
    assert destination.exists() and not source.exists()


def test_complete_orphan_mel_shard_is_resumed_without_recomputing(tmp_path, monkeypatch):
    import biosonic_cnn.mels as mel_module
    audio_root = tmp_path / "audio"
    path = audio_root / "compot1" / "short.ogg"
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(4000, dtype=np.float32), 32000, format="OGG", subtype="VORBIS")
    monkeypatch.setattr(mel_module.config, "AUDIO_ROOT", audio_root)
    cache = tmp_path / "mels"
    digest = mel_module.sha256_file(path)
    record = pd.DataFrame([{"filename": "compot1/short.ogg", "expected_size_bytes": path.stat().st_size,
                            "sha256": digest, "sample_count": 4000, "m4_partition": "internal_train",
                            "primary_label": "compot1", "source_sample_count": 4000, "segment_index": 0}])
    shard_dir, npy_path, meta_path = mel_module._shard_paths("compot1/short.ogg", cache)
    orphan = cache / f"{shard_dir.name}.tmp-old-run"
    orphan.mkdir(parents=True)
    np.save(orphan / "mels.npy", np.zeros((1, 128, 622), dtype=np.float16))
    (orphan / "metadata.json").write_text(json.dumps({"filename": "compot1/short.ogg", "source_sha256": digest,
        "config_hash": mel_module.mel_config_hash(), "segments": 1, "valid_frames": [12],
        "partition": "internal_train", "primary_label": "compot1", "shape": [1, 128, 622], "dtype": "float16"}))
    monkeypatch.setattr(mel_module, "compute_mel_window", lambda *args: (_ for _ in ()).throw(AssertionError("must reuse complete orphan")))
    resumed = mel_module.extract_recording_mels(record, cache)
    assert resumed["source_sha256"] == digest
    assert npy_path.exists() and meta_path.exists() and not orphan.exists()


def test_existing_mel_shard_source_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    import biosonic_cnn.mels as mel_module
    audio_root = tmp_path / "audio"
    path = audio_root / "compot1" / "short.ogg"
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(4000, dtype=np.float32), 32000, format="OGG", subtype="VORBIS")
    monkeypatch.setattr(mel_module.config, "AUDIO_ROOT", audio_root)
    record = pd.DataFrame([{"filename": "compot1/short.ogg", "expected_size_bytes": path.stat().st_size,
                            "sha256": mel_module.sha256_file(path), "sample_count": 4000,
                            "m4_partition": "internal_train", "primary_label": "compot1",
                            "source_sample_count": 4000, "segment_index": 0}])
    cache = tmp_path / "mels"
    mel_module.extract_recording_mels(record, cache)
    record.loc[0, "sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="SHA-256 changed"):
        mel_module.extract_recording_mels(record, cache)
