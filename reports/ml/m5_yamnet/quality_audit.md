# M5 YAMNet quality audit

Status: **PASS**

- Approved acquisition manifest: 3,373 unique files.
- Experiment set: 2,177 unique files; eight classes.
- M3 partitions: 1,748 train / 429 final holdout.
- Out-of-scope M2 samples in experiment: 0.
- Training/holdout group, author, filename, and SHA-256 overlaps: 0.
- Five grouped-CV folds passed class/stratum and leakage checks.
- Audio integrity: 2,177 files decoded and matched size, SHA-256, format, rate, channels, and frame counts.
- M5 embeddings map one-to-one to all 2,177 source recordings.
- Holdout evaluation was performed once after classifier freeze; it was not used for selection.

No M3/M4 model, report, manifest, or result was changed by the M5 workflow.
