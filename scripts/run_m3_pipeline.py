import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_ml.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="BioSonic M3 eight-class traditional ML workflow")
    parser.add_argument("--stage", choices=("validate", "features", "train-evaluate", "all"), default="all")
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
