# BioSonic publication checklist

This is the definitive proposed whitelist for the next update. It does not authorize staging. Stage only the reviewed paths after the final staged-content check. Every listed path was checked for existence and publication suitability during this review; rerun the checks immediately before staging.

## Definitive source, documentation, and aggregate-results whitelist

### Root and environment files

- `.gitignore`
- `LICENSE` — Apache-2.0 text; scope is original BioSonic source code as described in the README.
- `README.md`
- `requirements-main.txt`
- `requirements-yamnet.txt`

### Documentation and validation strategy

- `docs/audio_preprocessing_specification.md`
- `docs/publication_checklist.md`
- `reports/ml/validation_strategy.md` — historical 12-class plan, visibly superseded.
- `reports/ml/eight_class_validation_strategy.md` — current interim strategy.

### M3, M4, and M5 aggregate reports, tables, and model configuration

- `reports/ml/m3_eight_class/m3_eight_class_experiment_report.md` — canonical M3 aggregate report.
- `reports/ml/m3_eight_class/training_config.json` — class/model configuration and package versions; no source-recording IDs or dataset hashes.
- `reports/ml/m4_custom_cnn_experiment_report.md` — M4 aggregate report.
- `reports/ml/m4_custom_cnn/architecture.json` — architecture description.
- `reports/ml/m4_custom_cnn/internal_validation_metrics.json` — aggregate metrics.
- `reports/ml/m4_custom_cnn/holdout_metrics.json` — aggregate metrics.
- `reports/ml/m4_custom_cnn/m3_m4_model_comparison.csv` — aggregate model comparison.
- `models/m4_custom_cnn/training_config.json` — training configuration; no checkpoint included.
- `reports/ml/m5_yamnet/m5_yamnet_transfer_learning_report.md` — M5 aggregate report.
- `reports/ml/m5_yamnet/m3_m4_m5_comparison.csv` — aggregate comparison.
- `reports/ml/m5_yamnet/per_class_f1_comparison.csv` — per-class aggregate scores, not per-recording predictions.
- `reports/ml/m5_yamnet/quality_audit.md` — aggregate audit summary.
- `reports/ml/m5_yamnet/computational_performance.json` — aggregate timing and model-size summary.

### Model provenance and mapping metadata

- `models/m5_yamnet/PROVENANCE.md` — model, source, checkpoint, and AudioSet license notes.
- `models/m5_yamnet/source_provenance.json` — upstream source and pretrained-model checksums only; no BirdCLEF recording hashes.
- `models/m5_yamnet/class_mapping.json` — eight-class output mapping.
- `src/biosonic_yamnet/_official/LICENSE` — unchanged TensorFlow Models upstream notice.
- `src/biosonic_yamnet/_official/yamnet_class_map.csv` — AudioSet class map; carry its separate CC BY-SA 4.0 attribution and terms.

### Project source code

- `scripts/download_approved_audio.py`
- `scripts/run_m3_pipeline.py`
- `scripts/run_m4_cnn.py`
- `scripts/run_m5_yamnet.py`
- `src/biosonic_ml/__init__.py`
- `src/biosonic_ml/config.py`
- `src/biosonic_ml/dataset.py`
- `src/biosonic_ml/features.py`
- `src/biosonic_ml/io_utils.py`
- `src/biosonic_ml/pipeline.py`
- `src/biosonic_ml/splitting.py`
- `src/biosonic_ml/training.py`
- `src/biosonic_cnn/__init__.py`
- `src/biosonic_cnn/config.py`
- `src/biosonic_cnn/data.py`
- `src/biosonic_cnn/dataset.py`
- `src/biosonic_cnn/mels.py`
- `src/biosonic_cnn/model.py`
- `src/biosonic_cnn/pipeline.py`
- `src/biosonic_cnn/train.py`
- `src/biosonic_yamnet/__init__.py`
- `src/biosonic_yamnet/audio.py`
- `src/biosonic_yamnet/audit.py`
- `src/biosonic_yamnet/classifiers.py`
- `src/biosonic_yamnet/config.py`
- `src/biosonic_yamnet/dataset.py`
- `src/biosonic_yamnet/evaluation.py`
- `src/biosonic_yamnet/io_utils.py`
- `src/biosonic_yamnet/model.py`
- `src/biosonic_yamnet/reporting.py`
- `src/biosonic_yamnet/smoke.py`
- `src/biosonic_yamnet/workflow.py`
- `src/biosonic_yamnet/_official/features.py`
- `src/biosonic_yamnet/_official/params.py`
- `src/biosonic_yamnet/_official/yamnet.py`

### Tests

- `tests/test_download_approved_audio.py`
- `tests/test_m3_pipeline.py`
- `tests/test_m4_cnn.py`
- `tests/test_m5_yamnet.py`

## Existing public files and duplicate report

The first public commit already contains combined M2 figures, including `reports/audio_waveforms/all_six_waveforms.png`, `reports/features/mfcc/all_six_mfcc.png`, `reports/frequency_analysis/all_six_frequency_spectra.png`, `reports/spectrograms/mel/all_six_mel_spectrograms.png`, and `reports/spectrograms/stft/all_six_stft_spectrograms.png`. Visual inspection confirms the panels display original sample recording identifiers. These files are unchanged and are not part of the new-update staging set; do not delete, move, or untrack them here. The identifiers are already public, and their continued publication should be considered during the final manual review. README attribution applies to these dataset-derived figures.

The root-level `reports/ml/m3_eight_class_experiment_report.md` duplicates the nested canonical M3 report. Keep the nested report canonical. Exclude the root duplicate from staging; do not delete or modify it for cleanup in this update.

## Explicit exclusions

- All of `data/`, including audio, metadata, inventories, downloads, and derived datasets.
- `.venv/`, `.venv-yamnet/`, other environments, local package caches, and machine-specific configuration.
- Kaggle credentials, `kaggle.json`, `.env` files, tokens, keys, passwords, and other secrets.
- All model checkpoints and weights, including `.joblib`, `.pt`, `.pth`, `.h5`, and similar files; the local YAMNet checkpoint remains excluded.
- Fitted model artifacts, embedding caches, extraction shards, and output manifests.
- Download, split, segment-level, and recording-level manifests or inventories.
- Recording-level prediction CSVs and group/author metadata.
- Unreviewed metric JSON, per-recording CSVs, individual recording plots, generated figures, and temporary files.
- `reports/ml/download_summary.md` and `reports/ml/download_manifest.csv`.
- The root duplicate M3 report `reports/ml/m3_eight_class_experiment_report.md`.
- Any unlisted file, even if it is currently untracked.

## License and attribution review

- **Original BioSonic source code:** Apache License 2.0, as approved. The root `LICENSE` is the complete standard Apache-2.0 license text and applies only to original BioSonic source code. It does not automatically apply to BirdCLEF-derived reports/results, competition data, YAMNet model/checkpoint materials, or other third-party components; their separate terms remain in force.
- **Copyright-holder notice:** no copyright holder has been named in the source files or generic license text because the appropriate owner attribution was not established. Confirm whether a named notice is required and who may authorize it; do not invent one.
- **BirdCLEF+ 2026 data and derived reports/results:** competition data is subject to CC BY-NC-SA 4.0 and competition terms. When sharing derived reports/results, attribute BirdCLEF+ 2026 and the original recording contributors, identify and link the license, and indicate changes where applicable. No ownership claim is made. Review the NonCommercial and ShareAlike scope for aggregate reports/figures and public portfolio use before publication.
- **YAMNet source/model/checkpoint:** the official TensorFlow Models implementation and Google model listing document Apache-2.0. The direct checkpoint blob does not carry a separate license file, so its license is not independently verified; provenance notes its attribution basis in the model listing. Do not claim independent checkpoint-license verification, and keep checkpoint redistribution excluded.
- **YAMNet class map / AudioSet ontology:** CC BY-SA 4.0, distinct from AudioSet dataset CC BY 4.0 and from YAMNet model/source Apache-2.0. Preserve the class-map attribution and notice.
- Do not treat the root Apache license as a blanket license for every repository file.

## Security and large-file checks

Final preparation scan (2026-11-22) checked all 23 currently tracked files plus the 66-path proposed whitelist (82 unique text/binary files after overlap). No credential-pattern, email-address, or absolute local-path matches were found. No scanned file exceeded 10 MiB. Nine already tracked PNG figures were identified as binary; they are existing public combined M2 figures and remain unchanged. The proposed whitelist itself contains no binary files.

The source whitelist does contain a small number of BirdCLEF recording identifiers in functional constants: two inventory exclusions in the downloader/M3 config and the six M2 sample names used by M3/M4/M5 exclusion or smoke-test guards. These are not user credentials or per-recording result tables, but they are dataset-derived identifiers. The previously published combined M2 figures also display those sample identifiers. Review whether that limited exposure is acceptable before the public update. No recording hashes, author/group IDs, or recording-level predictions are included in the whitelist.

`.gitignore` is not a substitute for inspecting content or checking tracked files. Repeat the scan against the exact staged set.

Before staging, inspect every candidate for credentials, API keys, tokens, passwords, personal emails, local profile paths, dataset recording IDs, dataset hashes, author/group IDs, unexpected binary content, and files above 10 MB. Confirm `data/`, environments, checkpoints, caches, and temporary files are ignored and absent from staging.

## Final staged review (tomorrow)

- [ ] Reconfirm branch, remote, latest commit, complete `git status --short --untracked-files=all`, and zero pre-existing staged changes.
- [ ] Confirm every staged path belongs to the definitive whitelist above; stage only with explicit paths, never `git add .` or `git add -A`.
- [ ] Run `git diff --cached --name-only`, `git diff --cached --stat`, and `git diff --cached --check`.
- [ ] Inspect all staged text/code changes and verify the five already-public combined M2 figures are not modified or staged again.
- [ ] Re-scan staged files for secrets, personal data, absolute paths, recording IDs/hashes, and files over 10 MB.
- [ ] Run `git check-ignore -v` on representative data, environments, caches, credentials, and weights; ensure none are staged or tracked unexpectedly.
- [ ] Resolve the copyright-holder notice question and review dataset-derived material / non-commercial terms before public release.
- [ ] Obtain final user approval of the exact staged list before commit or push.
