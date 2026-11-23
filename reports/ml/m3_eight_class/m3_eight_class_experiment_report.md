# BioSonic M3 — Eight-class interim experiment

## Dataset attribution and licensing

This report contains aggregate results derived from BirdCLEF+ 2026 competition recordings and metadata. Attribute BirdCLEF+ 2026 and the original recording contributors; the competition dataset is subject to **CC BY-NC-SA 4.0** and the competition rules. BioSonic does not claim ownership of the source data or dataset-derived material. The root Apache-2.0 license applies to original BioSonic source code only, not dataset-derived reports or data.

## Scope and dataset

The intended M3 task has 12 bird species. At execution time, the authoritative manifest had 2,177 verified recordings; four approved labels (`swtman1`, `smbani`, `yecpar`, `spispi1`) had no verified local audio. This is an eight-class interim benchmark, not a 12-class result. The six M2 sample recordings outside the approved manifest were excluded. The other 1,196 approved recordings remain in the intended full acquisition set but were not a prerequisite for this run.

Working set: **2,177 recordings** across eight classes with 2,177 unique SHA-256 hashes. Every source was present, size-matched, decoded OGG/Vorbis, mono at 32 kHz, and its hash still matched after the run.

| Class | Recordings | Dataset % | Train | Validation | XC | iNat | Author groups | Unknown authors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compot1 | 368 | 16.90% | 294 | 74 | 185 | 183 | 183 | 32 |
| rufhor2 | 315 | 14.47% | 251 | 64 | 163 | 152 | 127 | 20 |
| saytan1 | 223 | 10.24% | 179 | 44 | 149 | 74 | 86 | 11 |
| crbthr1 | 268 | 12.31% | 213 | 55 | 156 | 112 | 116 | 23 |
| rutjac1 | 265 | 12.17% | 211 | 54 | 228 | 37 | 100 | 6 |
| rufnig1 | 256 | 11.76% | 215 | 41 | 112 | 144 | 90 | 1 |
| chobla1 | 244 | 11.21% | 195 | 49 | 160 | 84 | 122 | 5 |
| burowl | 238 | 10.93% | 190 | 48 | 130 | 108 | 133 | 11 |

## Split and leakage controls

A deterministic `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` generated the fresh holdout. Selected feasible fold 5 contains 1,748 training recordings and 429 validation recordings (19.71%). Stratification used `primary_label × collection`. Groups join normalized authors across species/collections and exact duplicate hashes. There is no recording, author-group, or content-group overlap. All 109 literal Unknown-author recordings remain in training. All eight labels and all 16 observed label×collection cells occur in both partitions. The earlier 676-recording 12-class holdout was not reused.

The validation partition holds out known authors; generalization to Unknown authors is unmeasured because all Unknown records remain in training. Author grouping does not establish spatial independence; coordinate grouping remains a future experiment.

## Segmentation and features

Audio was streamed using SoundFile at its native 32 kHz. Non-overlapping five-second windows yielded 17,273 complete windows and 2,153 short or final partial windows (19,426 total). Partial windows were right-padded with zeros; valid source samples, source offsets, and padding are explicit in the segment manifest. A tail shorter than the 1,024-sample FFT window is represented by one zero-padded frame. All source samples are retained; no segment audio was written.

Each recording has a fixed 204-dimensional vector. Per-window frame summaries use mean and standard deviation for 13 power-Mel MFCCs, first- and second-order deltas, spectral centroid, bandwidth, 85% rolloff, zero-crossing rate, RMS, and seven spectral-contrast bands. Per-window statistics are aggregated across the source recording with mean and standard deviation. Mel configuration: 32 kHz, Hann, n_fft/win_length 1,024, hop 256, 128 Slaney-scale (`htk=False`), Slaney-normalized power-Mel bands from 0–16 kHz; log power uses a 1e-10 floor and MFCC DCT-II orthonormal normalization. This is exploratory, not an optimized model configuration.

## Models and evaluation

Compared Logistic Regression, Random Forest, and RBF SVM in scikit-learn Pipelines. StandardScaler is inside the Logistic Regression and SVM pipelines; Random Forest is unscaled. Class weights are balanced for Logistic Regression/SVM and balanced-subsample for Random Forest. Five-fold StratifiedGroupKFold hyperparameter tuning used only the 1,748-record training partition. Scalers are fitted within each CV fold. Macro F1 is the selection metric. Every model and parameter choice was fixed before the final holdout scoring.

| Model | Mean training CV macro F1 | Holdout accuracy | Macro precision | Macro recall | Holdout macro F1 | Weighted F1 | OVR macro ROC-AUC | CV search seconds | Inference ms/record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svm | 0.6575 | 0.6573 | 0.6679 | 0.6400 | 0.6431 | 0.6548 | unsupported | 4.45 | 0.2669 |
| logistic_regression | 0.6550 | 0.6830 | 0.6857 | 0.6773 | 0.6769 | 0.6853 | 0.9244 | 3.01 | 0.0030 |
| random_forest | 0.6017 | 0.6434 | 0.6507 | 0.6260 | 0.6315 | 0.6394 | 0.9061 | 41.77 | 0.1506 |

Selected by training-CV macro F1: **svm**. Its final holdout macro F1 is 0.6431. Logistic Regression has a higher score on this one holdout; the holdout was not used to select it, and this difference is reported transparently.

### Selected model per-class holdout metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| compot1 | 0.6897 | 0.8108 | 0.7453 | 74 |
| rufhor2 | 0.7833 | 0.7344 | 0.7581 | 64 |
| saytan1 | 0.7111 | 0.7273 | 0.7191 | 44 |
| crbthr1 | 0.6167 | 0.6727 | 0.6435 | 55 |
| rutjac1 | 0.6800 | 0.6296 | 0.6538 | 54 |
| rufnig1 | 0.8000 | 0.3902 | 0.5246 | 41 |
| chobla1 | 0.5870 | 0.5510 | 0.5684 | 49 |
| burowl | 0.4754 | 0.6042 | 0.5321 | 48 |

Saved artifacts include per-model CV hyperparameter-grid results, final holdout metrics, per-class classification reports, confusion matrices (CSV and plot), per-class F1 plots, model comparison tables, all fitted estimators, and a separate best-model artifact. SVM has no calibrated probabilities, so its ROC-AUC is unsupported; Logistic Regression and Random Forest report multiclass one-vs-rest macro ROC-AUC.

## Limitations and next work

This is recording-level primary-label classification, not soundscape multi-label detection and not an animal-population estimate. A recording's primary label may be a weak label for every window. Results apply only to this current eight-class author-grouped subset. Per-class performance varies. No claim is made about unseen species or all BirdCLEF recordings.

Next, review class errors and confusion patterns, compare feature/segmentation settings using training-only grouped CV, and regenerate a 12-class split after the remaining approved recordings become available.

Reproduce with `.venv\Scripts\python.exe -m pytest tests\test_m3_pipeline.py tests\test_download_approved_audio.py -q`, followed by `scripts\run_m3_pipeline.py --stage validate`, `--stage features`, and `--stage train-evaluate`. Feature shards are resumable and checked against source SHA-256 and the extraction configuration.
