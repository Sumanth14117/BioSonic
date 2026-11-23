# BioSonic M6 — Training-only model optimization

## Dataset attribution

This report contains aggregate results derived from BirdCLEF+ 2026 recordings and metadata. Attribute BirdCLEF+ 2026 and the original recording contributors; competition data and applicable derived material are subject to CC BY-NC-SA 4.0 and competition terms. BioSonic does not claim ownership of the source data. The root Apache-2.0 license applies to original BioSonic source code only.

## Objective and scope

M6 evaluates whether classifier-side changes improve the frozen YAMNet embedding baseline. The scope remains the eight-class interim bird classification task; four classes from the original 12-class objective remain unavailable. No M3–M5 results or artifacts were changed.

## Existing model limitations and experiment choice

M3 uses 204-dimensional handcrafted recording summaries and already applies class weighting to Logistic Regression/SVM; M6 tests whether its regularization settings transfer to the frozen-embedding representation. M4 uses five-second power-Mel windows, pads incomplete tails, assigns the source recording label to its windows, and averages segment probabilities at recording level. Its internal-validation macro-F1 (0.7603) and historical shared-holdout score are from different partitions and the latter is not valid for new tuning. M5 uses frozen 1,024-dimensional YAMNet embeddings and had a post-resampling waveform range diagnostic outside [-1, 1]. M6 does not tune on the reused holdout and does not alter any M3–M5 audio inputs. It focuses on class weighting and classifier regularization on M5's existing training-only embedding cache. CNN augmentation, CNN architecture changes, and new YAMNet waveform handling were not investigated in this run; those changes would need new leakage-safe training-side validation, with partial-window label noise handled explicitly.

## Dataset, partitions, and leakage controls

M6 used **1,748 training recordings** and eight classes. Inputs were the existing per-recording mean-pooled YAMNet embeddings (`1024` dimensions), aligned to the training prefix of the authoritative split manifest. The manifest reader stops at the first non-training row; no holdout labels, embeddings, predictions, or metrics were read. The old 429-recording holdout was not used.

Validation used five-fold `StratifiedGroupKFold` (seed 42), stratified on `primary_label × collection`, grouped by the existing author/content `group_id`. Each fold was checked for zero group overlap, all eight classes on both sides, and all observed label-by-collection strata on both sides. All scaling is inside the model pipeline and is fit separately on each fold's training data.

## Improvements investigated

Fifteen prespecified candidates compared Logistic Regression with balanced or unweighted loss across C values 0.01, 0.1, 1, and 10; RBF SVM with balanced or unweighted loss across C values 0.5, 1, and 5; and one class-balanced Random Forest configuration. The exact candidate parameters and fold scores are in `candidate_summary.csv` and `fold_metrics.csv`. Macro-F1 was the sole model-selection metric. No threshold tuning or probability calibration was performed.

## Training-side validation results

### Top candidates

| Model | Parameters | CV macro-F1 mean ± SD | Accuracy | Macro precision | Macro recall | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | `{"C": 0.1, "class_weight": "balanced", "max_iter": 4000, "random_state": 42}` | 0.5972 ± 0.0517 | 0.5996 | 0.6087 | 0.6026 | 0.5959 |
| logistic_regression | `{"C": 0.1, "class_weight": null, "max_iter": 4000, "random_state": 42}` | 0.5925 ± 0.0587 | 0.5974 | 0.6080 | 0.5972 | 0.5915 |
| logistic_regression | `{"C": 0.01, "class_weight": "balanced", "max_iter": 4000, "random_state": 42}` | 0.5846 ± 0.0421 | 0.5900 | 0.6001 | 0.5936 | 0.5851 |
| logistic_regression | `{"C": 0.01, "class_weight": null, "max_iter": 4000, "random_state": 42}` | 0.5815 ± 0.0439 | 0.5914 | 0.6032 | 0.5893 | 0.5829 |
| rbf_svm | `{"C": 5.0, "class_weight": "balanced", "decision_function_shape": "ovr", "gamma": "scale", "kernel": "rbf", "probability": false, "random_state": 42}` | 0.5661 ± 0.0492 | 0.5679 | 0.5817 | 0.5721 | 0.5661 |
| logistic_regression | `{"C": 1.0, "class_weight": "balanced", "max_iter": 4000, "random_state": 42}` | 0.5595 ± 0.0532 | 0.5635 | 0.5719 | 0.5646 | 0.5592 |
| logistic_regression | `{"C": 1.0, "class_weight": null, "max_iter": 4000, "random_state": 42}` | 0.5573 ± 0.0542 | 0.5629 | 0.5727 | 0.5625 | 0.5571 |
| rbf_svm | `{"C": 5.0, "class_weight": null, "decision_function_shape": "ovr", "gamma": "scale", "kernel": "rbf", "probability": false, "random_state": 42}` | 0.5571 ± 0.0514 | 0.5628 | 0.5795 | 0.5631 | 0.5575 |
| logistic_regression | `{"C": 10.0, "class_weight": "balanced", "max_iter": 4000, "random_state": 42}` | 0.5385 ± 0.0508 | 0.5415 | 0.5531 | 0.5434 | 0.5384 |
| logistic_regression | `{"C": 10.0, "class_weight": null, "max_iter": 4000, "random_state": 42}` | 0.5384 ± 0.0508 | 0.5427 | 0.5511 | 0.5434 | 0.5388 |

Selected by training-side mean macro-F1: **logistic_regression** with `{"C": 0.1, "class_weight": "balanced", "max_iter": 4000, "random_state": 42}`. Its cross-validated out-of-fold macro-F1 is 0.5929; accuracy 0.5990, macro precision 0.5992, macro recall 0.5920, weighted F1 0.5981.

### Selected out-of-fold per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| compot1 | 0.6058 | 0.7109 | 0.6541 | 294 |
| rufhor2 | 0.7042 | 0.6733 | 0.6884 | 251 |
| saytan1 | 0.6444 | 0.6480 | 0.6462 | 179 |
| crbthr1 | 0.5176 | 0.4836 | 0.5000 | 213 |
| rutjac1 | 0.6083 | 0.6256 | 0.6168 | 211 |
| rufnig1 | 0.6839 | 0.4930 | 0.5730 | 215 |
| chobla1 | 0.4976 | 0.5282 | 0.5124 | 195 |
| burowl | 0.5317 | 0.5737 | 0.5519 | 190 |

### Selected out-of-fold confusion matrix

| Actual / Predicted | compot1 | rufhor2 | saytan1 | crbthr1 | rutjac1 | rufnig1 | chobla1 | burowl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compot1 | 209 | 5 | 6 | 5 | 13 | 28 | 8 | 20 |
| rufhor2 | 4 | 169 | 22 | 9 | 8 | 1 | 15 | 23 |
| saytan1 | 7 | 13 | 116 | 11 | 11 | 0 | 14 | 7 |
| crbthr1 | 7 | 9 | 11 | 103 | 21 | 7 | 36 | 19 |
| rutjac1 | 10 | 9 | 10 | 17 | 132 | 4 | 18 | 11 |
| rufnig1 | 83 | 6 | 0 | 5 | 4 | 106 | 3 | 8 |
| chobla1 | 7 | 3 | 11 | 34 | 23 | 6 | 103 | 8 |
| burowl | 18 | 26 | 4 | 15 | 5 | 3 | 10 | 109 |

## Comparison with M3–M5

M6's primary estimate is grouped five-fold CV on the 1,748-recording training partition. The best M6 classifier (balanced Logistic Regression, C=0.1; mean macro-F1 0.5972 ± 0.0517) matches the prior M5 classifier baseline and does not improve it. The prior M5 result is 0.5972 ± 0.0517 under the same training-only grouped-CV protocol. Existing M3 training-CV results were 0.6575 ± 0.0397 for RBF SVM and 0.6550 ± 0.0420 for Logistic Regression on different handcrafted features; M3 remains the strongest comparable training-CV result among those reported. The M4 internal-validation macro-F1 of 0.7603 used its distinct 1,444/304 split and is not directly comparable. Previously reported M3–M5 holdout metrics are intentionally not repeated as M6 selection evidence and were not accessed.

The final refit ran on CPU (scikit-learn estimators; GPU not used); the available RTX 4060 was detected but not used. Candidate CV plus final fitting took 80.4 seconds; final refit took 8.6 seconds. Environment: Python 3.14.7, scikit-learn 1.9.1, NumPy 2.5.5.

## Selection and limitations

The selected estimator is refit on all 1,748 training recordings after grouped CV. The selected fold mean and selected-candidate out-of-fold table use the same folds; both are model-selection results and may be optimistic, not an independent unbiased estimate. Use a new author/spatially separated dataset for confirmation. This remains an eight-class interim, recording-level single-label experiment. Weak primary labels, unknown-author generalization, geographic dependence, shared historical M3–M5 holdout reuse, and the M5 input-range concern remain. M6 does not establish performance for the four unavailable classes or soundscape multi-label detection.

## Reproducibility

Run from the repository root: `.venv\Scripts\python.exe scripts\run_m6_optimization.py`. The default output directories are write-once and the script refuses to overwrite them. For an additional independent run with the same fixed configuration, use `--run-id run_02` (or another unique safe identifier) to write under `reports/ml/m6_optimization/runs/<run-id>/` and `models/m6_optimization/runs/<run-id>/`. Configuration, fold-level aggregate metrics, selected out-of-fold per-class metrics, confusion matrix, and the selected classifier with SHA-256 metadata are saved. No per-recording prediction file is written.

## M7 next steps

Keep M7 focused on explainability and biodiversity analytics with the validation boundary intact. Explain model behavior using training-side folds, document uncertainty and species/collection limitations, and design ecological activity summaries that are not presented as direct population estimates.
