"""M5 paths and fixed experiment choices."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_MANIFEST = ROOT / "reports/ml/eight_class_split_manifest.csv"
DOWNLOAD_MANIFEST = ROOT / "reports/ml/download_manifest.csv"
AUDIO_ROOT = ROOT / "data/raw/train_audio"
TRAIN_CSV = ROOT / "data/metadata/train.csv"

OFFICIAL_SOURCE_COMMIT = "c14bf9ad91962cf189f9f58db2132c06247fcd53"
OFFICIAL_SOURCE_BASE = (
    "https://github.com/tensorflow/models/tree/"
    + OFFICIAL_SOURCE_COMMIT
    + "/research/audioset/yamnet"
)
WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
WEIGHTS_PATH = ROOT / "models/m5_yamnet/yamnet.h5"
WEIGHTS_BYTES = 15_296_092
WEIGHTS_MD5 = "8234225c045859752bd4a2e101f8cd1a"
WEIGHTS_SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"
OFFICIAL_SOURCE_SHA256 = {
    "params.py": "925bb1e62461016031f98aea09aeac28975dd516f5747513767de5d1b06b6145",
    "features.py": "e6cd53f81d072c7c43be4c7fff2b9dd0c5ccc7d64f2fbcfc85b44013d6d2ed5e",
    "yamnet.py": "7ef3df32b7ecb782490b5a04d7581a23cfcf701dbf476bc4d03deefe22cdb040",
    "yamnet_class_map.csv": "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
    "LICENSE": "5b17814bf0de8cf65069bc6d7cc38cff19fcaa864d243423ad3ef3db01b52385",
}

OFFICIAL_SOURCE_DIR = ROOT / "src/biosonic_yamnet/_official"
MODEL_ROOT = ROOT / "models/m5_yamnet"
DATA_ROOT = ROOT / "data/processed/m5_yamnet"
TRAIN_SHARDS = DATA_ROOT / "training_shards_v2"
HOLDOUT_SHARDS = DATA_ROOT / "holdout_shards_v2"
REPORT_ROOT = ROOT / "reports/ml/m5_yamnet"

WORKING_LABELS = (
    "compot1", "rufhor2", "saytan1", "crbthr1",
    "rutjac1", "rufnig1", "chobla1", "burowl",
)
SEED = 42
SAMPLE_RATE = 32_000
MODEL_SAMPLE_RATE = 16_000
TRAIN_RECORDINGS = 1_748
HOLDOUT_RECORDINGS = 429
EMBEDDING_DIM = 1_024
N_SPLITS = 5
STRATIFICATION = "primary_label × collection"
AGGREGATION = "arithmetic mean of all official YAMNet patch embeddings per recording"

SAMPLE_AUDIO = ROOT / "data/raw/train_audio/rubthr1/XC1006659.ogg"
SAMPLE_AUDIO_NAME = "rubthr1/XC1006659.ogg"
M2_SAMPLE_FILES = {
    "rubthr1/XC1006659.ogg", "banana/XC1034681.ogg",
    "fepowl/XC1034537.ogg", "bafcur1/XC1018568.ogg",
    "22973/XC892927.ogg", "244024/iNat103300.ogg",
}

EXPERIMENT_VERSION = "biosonic-m5-yamnet-mean-embedding-v2-official-pcm16-input"
