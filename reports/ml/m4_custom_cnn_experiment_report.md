# M4 Custom CNN — eight-class interim experiment

## Dataset attribution and licensing

This report contains aggregate results derived from BirdCLEF+ 2026 competition recordings and metadata. Attribute BirdCLEF+ 2026 and the original recording contributors; the competition dataset is subject to **CC BY-NC-SA 4.0** and the competition rules. BioSonic does not claim ownership of the source data or dataset-derived material. The root Apache-2.0 license applies to original BioSonic source code only, not dataset-derived reports or data.

## Scope and existing benchmark

This experiment continues the eight-class interim M3 scope using exactly the 2,177 approved recordings. The four unavailable approved species (`swtman1`, `smbani`, `yecpar`, `spispi1`) remain excluded; this is not a 12-class result. The original M3 recording split was preserved: 1,748 M3 training recordings and the same 429-recording holdout. M3 has already evaluated that holdout, so CNN scores below are a holdout evaluation on the existing benchmark, not a newly untouched external test.

## Internal validation split

A deterministic author/content-grouped `StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)` subdivided only the 1,748 M3 training recordings. The selected feasible fold was 2: **1,444 internal training**, **304 internal validation** (17.39%), with the original **429 holdout** left out of all training, early stopping, and tuning. All eight classes and all 16 label×collection strata were represented internally on both sides. There was no author/content-group overlap; all 109 literal Unknown-author training recordings remain in internal training.

## Mel representation and preprocessing

One spectrogram is generated for each of the 19,426 windows in the M3 segment manifest. The cache contains log-power values as float16 (disk cache); values are converted to float32 and normalized in the Dataset. Configuration: 32 kHz mono, five-second / 160,000-sample non-overlapping windows, `n_fft=win_length=1024`, `hop_length=256`, periodic Hann, 128 Slaney-scale Slaney-normalized Mel bands, 0–16 kHz, no centered padding. Incomplete/short tails are right-zero-padded and carry valid sample/frame counts. The extractor uses the same 128-band Slaney power-Mel definition as M3; it uses an explicit `10*log10(max(power, 1e-10))` transform and periodic Hann consistent with M3's feature code. The persisted M3 exploratory plots used a distinct visualization convention, so no claim of bitwise equality to their plotted matrices is made.

Per-band mean and standard deviation are fit only on valid frames in the internal-training recordings. Invalid padded time frames are set to zero after normalization. The final holdout contributes no normalization statistics.

## Custom architecture

The model is trained from scratch with no pretrained weights. Four blocks each contain two 3×3 Conv2D–BatchNorm–ReLU layers, 2× max pooling, and Dropout2D. Channels grow 1→16→32→64→128. Adaptive average pooling produces one 128-value vector, followed by a 128-unit ReLU/dropout layer and eight raw logits trained with CrossEntropyLoss.

Trainable parameters: **311,256**. Model input is `[batch, 1, 128, 622]`; output is `[batch, 8]`.

## Training configuration

Device: **cuda** (NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB reported VRAM). Batch size 32; AdamW (lr=0.0003, weight decay=0.0001); ReduceLROnPlateau on internal-validation recording macro F1; up to 30 epochs with early stopping patience 6. Seed 42 and deterministic cuDNN settings. CUDA mixed precision is enabled when CUDA is available. A weighted segment sampler gives equal total sampling weight per class and per recording, preventing long recordings or large classes from dominating the minibatch stream.

Training stopped at epoch 30; best epoch **27**. Best-epoch training loss: 0.5333; sampled recording-aggregate training macro F1: 0.9601; internal-validation segment loss: 5.0721; internal-validation recording macro F1: 0.7603. Training duration: 1726.7 seconds. Peak CUDA allocated memory: 801.9 MiB (840,826,368 bytes).

Training loss is segment-sampled; train macro F1 aggregates sampled segment probabilities by recording. Validation and holdout losses are segment-level diagnostics, while their headline metrics average segment probabilities per recording before scoring so each recording counts once.

## Recording-level results

CNN segment probabilities were averaged across all windows from each recording. The resulting recording-level scores are:

| Partition | Recordings | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | OVR macro ROC-AUC | Inference time | ms / recording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal validation | 304 | 0.7237 | 0.8195 | 0.7769 | 0.7603 | 0.7111 | 0.9331 | 7.39s | 24.315 |
| Existing M3 holdout | 429 | 0.8788 | 0.8913 | 0.8692 | 0.8726 | 0.8768 | 0.9851 | 6.19s | 14.420 |

### Comparison with traditional ML

The M3 models use their existing five-fold training-CV mean macro F1 for model selection; the CNN uses internal-validation macro F1 for early stopping. All holdout columns below refer to the same previously evaluated M3 429 recordings.

| Model | Selection metric value | Holdout accuracy | Holdout macro F1 | Holdout weighted F1 | Holdout OVR ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 logistic_regression | CV macro-F1 0.6550 | 0.6830 | 0.6769 | 0.6853 | 0.9244 |
| M3 random_forest | CV macro-F1 0.6017 | 0.6434 | 0.6315 | 0.6394 | 0.9061 |
| M3 svm | CV macro-F1 0.6575 | 0.6573 | 0.6431 | 0.6548 | unsupported |
| M4 custom CNN | Internal validation macro-F1 0.7603 | 0.8788 | 0.8726 | 0.8768 | 0.9851 |

The different selection scores are not directly interchangeable. No unsupported claim that the CNN is better is made. The shared holdout has previously influenced M3 development, so the M4 holdout comparison is benchmark reuse rather than independent confirmation.

## Limitations and next work

The per-window target is inherited from each recording's primary label and may be weak for silent or mixed-species windows. Padding may affect CNN predictions despite the explicit frame mask and neutral normalized padding. Unknown authors remain training-only, so generalization to Unknown contributors cannot be assessed. Author grouping does not ensure spatial independence. This is recording-level primary-label classification, not soundscape multi-label detection or a direct population estimate. Results apply only to the current eight-class subset.

## Reproducibility

Key outputs are stored under `reports/ml/m4_custom_cnn/`: `architecture.json`, `cnn_architecture.png`, `training_validation_loss.png`, `training_validation_macro_f1.png`, `holdout_confusion_matrix.png`, `per_class_holdout_f1_comparison.png`, `m3_m4_holdout_comparison.png`, the internal-validation and holdout classification/confusion/metric reports, and `sample_mel_spectrograms.png`. The best checkpoint is `models/m4_custom_cnn/best_model.pt`; the final resumable checkpoint, configuration, and epoch history are alongside it. Training ended at epoch 30 with the selected best checkpoint from epoch 27.

The per-class F1 and confusion figures are rendered from persisted recording-level evaluation CSVs; they do not run inference again.

`.venv\Scripts\python.exe -m pytest tests\test_m4_cnn.py tests\test_m3_pipeline.py tests\test_download_approved_audio.py -q`

`.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage prepare`

`.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage mels`

`.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage train`

To resume an interrupted training run, add `--resume` to the train command. To resume feature generation, rerun the mels stage; completed atomic per-recording shards are checked against source SHA-256 and the feature configuration. No model or feature output should be manually relabeled or edited.
