import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config


class DatasetError(ValueError):
    pass


def normalize_author(value) -> str:
    if pd.isna(value) or not str(value).strip() or str(value).strip().casefold() == "unknown":
        return "__unknown__"
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _union_groups(rows: pd.DataFrame) -> dict[str, str]:
    parent = {name: name for name in rows.filename}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, subset in rows.groupby("author_group_key", sort=False):
        names = subset.filename.tolist()
        for name in names[1:]:
            union(names[0], name)
    for _, subset in rows.groupby("sha256", sort=False):
        names = subset.filename.tolist()
        for name in names[1:]:
            union(names[0], name)
    result = {}
    for name in parent:
        root = find(name)
        result[name] = hashlib.sha256(root.encode("utf-8")).hexdigest()[:20]
    return result


def load_working_dataset(manifest_path=config.MANIFEST, train_path=config.TRAIN_CSV,
                         taxonomy_path=config.TAXONOMY_CSV, inventory_path=config.INVENTORY_CSV,
                         audio_root=config.AUDIO_ROOT,
                         expected_count=2177, decode=True) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    train = pd.read_csv(train_path)
    taxonomy = pd.read_csv(taxonomy_path)
    inventory = pd.read_csv(inventory_path)
    required_m = {"filename", "primary_label", "verification_status", "expected_size_bytes"}
    required_t = {"filename", "primary_label", "author", "collection", "class_name"}
    if not required_m.issubset(manifest.columns) or not required_t.issubset(train.columns):
        raise DatasetError("Manifest or training metadata is missing required columns")
    if manifest.filename.duplicated().any():
        raise DatasetError("Approved manifest contains duplicate filenames")
    if train.filename.duplicated().any():
        raise DatasetError("Training metadata contains duplicate filenames")
    if not set(config.WORKING_LABELS).issubset(set(taxonomy.primary_label)):
        raise DatasetError("A working label is missing from taxonomy")
    tax_rows = taxonomy.set_index("primary_label")
    for label in config.WORKING_LABELS:
        if str(tax_rows.loc[label, "class_name"]) != "Aves":
            raise DatasetError(f"Taxonomy class mismatch for {label}")
    if not {"FileName", "SizeBytes"}.issubset(inventory.columns):
        raise DatasetError("Audio inventory is missing FileName or SizeBytes")
    inv = inventory.set_index("FileName")["SizeBytes"]
    approved = manifest[manifest.primary_label.isin(config.ALL_APPROVED_LABELS)].copy()
    verified = approved[approved.verification_status.eq("verified")].copy()
    metadata = train.set_index("filename", drop=False)
    records = []
    seen_hashes = {}
    for row in verified.itertuples(index=False):
        filename = str(row.filename).replace("\\", "/")
        if filename in config.EXCLUDED_FILES or filename in config.SAMPLE_FILES:
            continue
        label = str(row.primary_label)
        if label not in config.WORKING_LABELS:
            continue
        if filename not in metadata.index:
            raise DatasetError(f"No metadata row for approved file {filename}")
        meta = metadata.loc[filename]
        if str(meta.primary_label) != label or str(meta.class_name) != "Aves":
            raise DatasetError(f"Metadata label/class mismatch for {filename}")
        path = Path(audio_root).joinpath(*filename.split("/"))
        if not path.is_file():
            raise DatasetError(f"Verified file is absent: {filename}")
        expected = int(row.expected_size_bytes)
        inventory_key = "train_audio/" + filename
        if inventory_key not in inv.index or int(inv.loc[inventory_key]) != expected:
            raise DatasetError(f"Inventory missing or size disagrees for {filename}")
        actual = path.stat().st_size
        if actual != expected:
            raise DatasetError(f"File-size mismatch for {filename}: expected {expected}, got {actual}")
        digest = sha256_file(path)
        try:
            info = sf.info(str(path))
            if decode:
                audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
                if sr != config.SR or audio.shape[1] != 1 or not np.isfinite(audio).all():
                    raise DatasetError(f"Audio validation failed for {filename}")
        except Exception as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"Cannot decode {filename}: {exc}") from exc
        if int(info.samplerate) != config.SR or info.channels != 1:
            raise DatasetError(f"Expected mono {config.SR} Hz audio: {filename}")
        if info.format != "OGG" or info.subtype != "VORBIS":
            raise DatasetError(f"Expected OGG/Vorbis source: {filename}")
        if digest in seen_hashes:
            # Retain the record; duplicate hashes are joined into one split group.
            pass
        seen_hashes[digest] = filename
        records.append({"filename": filename, "primary_label": label, "collection": str(meta.collection),
                        "author": str(meta.author), "author_group_key": normalize_author(meta.author),
                        "sha256": digest, "expected_size_bytes": expected, "actual_size_bytes": actual,
                        "sample_rate": int(info.samplerate), "channels": int(info.channels),
                        "sample_count": int(info.frames), "duration_seconds": float(info.duration),
                        "audio_path": str(path)})
    frame = pd.DataFrame(records)
    if len(frame) != expected_count:
        raise DatasetError(f"Expected {expected_count} working recordings, found {len(frame)}")
    if set(frame.primary_label) != set(config.WORKING_LABELS):
        raise DatasetError("Working dataset does not contain exactly the eight configured classes")
    groups = _union_groups(frame)
    frame["group_id"] = frame.filename.map(groups)
    frame["author_group_id"] = frame.author_group_key.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:20])
    return frame.sort_values(["primary_label", "filename"], kind="stable").reset_index(drop=True)
