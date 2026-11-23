from pathlib import Path

from biosonic_ml.config import ROOT, WORKING_LABELS, SR, SEGMENT_SECONDS, SEGMENT_SAMPLES, SEED

M3_SPLIT = ROOT / "reports/ml/eight_class_split_manifest.csv"
M3_SEGMENTS = ROOT / "reports/ml/eight_class_segment_manifest.csv"
M3_REPORT_DIR = ROOT / "reports/ml/m3_eight_class"
M3_REPORT = ROOT / "reports/ml/m3_eight_class_experiment_report.md"
M3_FEATURE_ROOT = ROOT / "data/processed/m3_eight_class_v2"
M4_DATA_ROOT = ROOT / "data/processed/m4_mels"
M4_SPLIT = ROOT / "reports/ml/m4_internal_split_manifest.csv"
M4_SEGMENTS = ROOT / "reports/ml/m4_segment_manifest.csv"
M4_SPLIT_SUMMARY = ROOT / "reports/ml/m4_internal_split_summary.json"
M4_MODEL_ROOT = ROOT / "models/m4_custom_cnn"
M4_REPORT_DIR = ROOT / "reports/ml/m4_custom_cnn"
M4_REPORT = ROOT / "reports/ml/m4_custom_cnn_experiment_report.md"
AUDIO_ROOT = ROOT / "data/raw/train_audio"
DOWNLOAD_MANIFEST = ROOT / "reports/ml/download_manifest.csv"
TAXONOMY_CSV = ROOT / "data/metadata/taxonomy.csv"

EXPECTED_RECORDINGS = 2177
EXPECTED_CLASS_COUNTS = {"compot1": 368, "rufhor2": 315, "saytan1": 223, "crbthr1": 268,
                         "rutjac1": 265, "rufnig1": 256, "chobla1": 244, "burowl": 238}
N_FFT = WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 0
FMAX = 16000
MEL_VERSION = "m4-log-power-mel-slaney-periodic-hann-v1"
INTERNAL_SPLITS = 6
INTERNAL_TARGET = 0.175
SELECTION_METRIC = "recording-level macro F1"
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(WORKING_LABELS)}
INDEX_TO_LABEL = tuple(WORKING_LABELS)
