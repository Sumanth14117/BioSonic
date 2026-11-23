from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .data import sha_key


class MelWindowDataset(Dataset):
    """One item per fixed five-second window; recording identity is retained."""

    def __init__(self, index: pd.DataFrame, normalizer: dict, cache_root=config.M4_DATA_ROOT / "mels"):
        required = {"filename", "primary_label", "m4_partition", "segment_index", "valid_frames", "sha256",
                    "author_group_id", "collection", "source_sample_count", "padded_sample_count"}
        if not required.issubset(index.columns):
            raise ValueError(f"Segment index is missing columns: {sorted(required - set(index.columns))}")
        if normalizer.get("fit_partition") != "internal_train":
            raise ValueError("CNN input normalization must be fit on internal training only")
        self.index = index.reset_index(drop=True)
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(config.N_MELS, 1)
        self.std = np.asarray(normalizer["std"], dtype=np.float32).reshape(config.N_MELS, 1)
        if self.mean.shape != (config.N_MELS, 1) or self.std.shape != (config.N_MELS, 1) or np.any(self.std <= 0):
            raise ValueError("Malformed Mel normalization vectors")
        self.cache_root = Path(cache_root)
        self._open_shards = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _shard(self, filename):
        if filename in self._open_shards:
            self._open_shards.move_to_end(filename)
            return self._open_shards[filename]
        path = self.cache_root / sha_key(filename) / "mels.npy"
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        self._open_shards[filename] = arr
        while len(self._open_shards) > 64:
            self._open_shards.popitem(last=False)
        return arr

    def __getitem__(self, index):
        row = self.index.iloc[index]
        filename = str(row.filename)
        frame = int(row.segment_index)
        valid_frames = int(row.valid_frames)
        raw = np.asarray(self._shard(filename)[frame], dtype=np.float32)
        if raw.shape != (config.N_MELS, 622) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid cached Mel shape/content: {filename}#{frame}")
        values = (raw - self.mean) / self.std
        values[:, valid_frames:] = 0.0
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite normalized Mel features: {filename}#{frame}")
        label = config.LABEL_TO_INDEX[str(row.primary_label)]
        return {"x": torch.from_numpy(values[None].copy()), "y": torch.tensor(label, dtype=torch.long),
                "filename": filename, "segment_index": torch.tensor(frame, dtype=torch.long),
                "m4_partition": str(row.m4_partition)}
