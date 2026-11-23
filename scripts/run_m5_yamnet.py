"""Run the isolated BioSonic M5 YAMNet experiment in explicit safe stages."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_yamnet import config
from biosonic_yamnet.audit import run_quality_audit
from biosonic_yamnet.evaluation import evaluate_frozen_holdout
from biosonic_yamnet.reporting import create_final_reports
from biosonic_yamnet.smoke import run_smoke_test
from biosonic_yamnet.workflow import extract_training_stage, train_and_freeze_stage


def run_stage(name: str):
    print(f"\n=== M5 stage: {name} ===", flush=True)
    if name == "smoke":
        result = run_smoke_test(output_path=config.REPORT_ROOT / "model_smoke.json")
    elif name == "extract-train":
        result = extract_training_stage()
    elif name == "train-freeze":
        result = train_and_freeze_stage()
    elif name == "evaluate-holdout":
        result = evaluate_frozen_holdout()
    elif name == "audit-report":
        audit = run_quality_audit()
        report = create_final_reports(json.loads((config.REPORT_ROOT / "holdout_metrics.json").read_text(encoding="utf-8")))
        result = {"audit_status": audit["status"], **report}
    else:
        raise ValueError(name)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["smoke", "extract-train", "train-freeze", "evaluate-holdout", "audit-report", "all"])
    args = parser.parse_args()
    if args.stage == "all":
        run_stage("smoke")
        run_stage("extract-train")
        run_stage("train-freeze")
        run_stage("evaluate-holdout")
        run_stage("audit-report")
    else:
        run_stage(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
