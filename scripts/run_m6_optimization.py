"""Run the M6 training-only classifier optimization."""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_ml.m6_optimization import run_from_project  # noqa: E402


if __name__ == "__main__":
    from biosonic_ml.m6_optimization import M6_MODEL_ROOT, M6_REPORT_ROOT  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Write a separate run under reports/models M6 runs directories")
    args = parser.parse_args()
    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.run_id):
            parser.error("--run-id must be 1-40 letters, digits, underscores, or hyphens")
        report_root = M6_REPORT_ROOT / "runs" / args.run_id
        model_root = M6_MODEL_ROOT / "runs" / args.run_id
    else:
        report_root, model_root = M6_REPORT_ROOT, M6_MODEL_ROOT
    result = run_from_project(report_root=report_root, model_root=model_root)
    summary = result["summary"]
    print(f"Selected {summary['selected_model']} by grouped training CV macro-F1 "
          f"{summary['selected_cv']['mean_macro_f1']:.4f} ± "
          f"{summary['selected_cv']['std_macro_f1']:.4f}")
    print(f"Report: {report_root / 'm6_optimization_report.md'}")
