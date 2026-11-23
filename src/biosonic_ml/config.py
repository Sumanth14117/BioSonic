from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports/ml/download_manifest.csv"
TRAIN_CSV = ROOT / "data/metadata/train.csv"
TAXONOMY_CSV = ROOT / "data/metadata/taxonomy.csv"
INVENTORY_CSV = ROOT / "data/metadata/birdclef_file_inventory.csv"
AUDIO_ROOT = ROOT / "data/raw/train_audio"
SPLIT_MANIFEST = ROOT / "reports/ml/eight_class_split_manifest.csv"
SPLIT_SUMMARY = ROOT / "reports/ml/eight_class_split_summary.json"
STRATEGY_REPORT = ROOT / "reports/ml/eight_class_validation_strategy.md"
FEATURE_ROOT = ROOT / "data/processed/m3_eight_class_v2"
FEATURES_NPZ = FEATURE_ROOT / "features.npz"
FEATURES_CSV = FEATURE_ROOT / "feature_metadata.csv"
FEATURE_CONFIG = FEATURE_ROOT / "feature_config.json"
MODEL_ROOT = ROOT / "models/m3_eight_class"
REPORT_ROOT = ROOT / "reports/ml/m3_eight_class"
FINAL_REPORT = ROOT / "reports/ml/m3_eight_class_experiment_report.md"

ALL_APPROVED_LABELS = ("compot1", "swtman1", "rufhor2", "smbani", "saytan1", "crbthr1",
                       "rutjac1", "rufnig1", "chobla1", "yecpar", "spispi1", "burowl")
WORKING_LABELS = ("compot1", "rufhor2", "saytan1", "crbthr1", "rutjac1", "rufnig1", "chobla1", "burowl")
EXCLUDED_FILES = {"swtman1/XC299145.ogg", "crbthr1/XC670199.ogg"}
SAMPLE_FILES = {"rubthr1/XC1006659.ogg", "banana/XC1034681.ogg", "fepowl/XC1034537.ogg",
                "bafcur1/XC1018568.ogg", "22973/XC892927.ogg", "244024/iNat103300.ogg"}
SR = 32000
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SR * SEGMENT_SECONDS
SEED = 42
N_SPLITS = 5
SPLIT_TARGET = 0.20
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 128
N_MFCC = 13

FEATURE_NAMES = tuple(
    [f"mfcc_{i}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")] +
    [f"delta_mfcc_{i}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")] +
    [f"delta2_mfcc_{i}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")] +
    [f"{name}_{stat}" for name in ("centroid", "bandwidth", "rolloff", "zcr", "rms", "contrast_0", "contrast_1", "contrast_2", "contrast_3", "contrast_4", "contrast_5", "contrast_6") for stat in ("mean", "std")]
)
FEATURE_VERSION = "m3-eight-class-power-mel-mfcc-delta-recording-aggregate-v2"
