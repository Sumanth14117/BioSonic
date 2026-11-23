import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_cnn.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="BioSonic M4 custom CNN training/evaluation")
    parser.add_argument("--stage", choices=("prepare", "mels", "train", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted CNN training checkpoint")
    args = parser.parse_args()
    run(args.stage, resume=args.resume)


if __name__ == "__main__":
    main()
