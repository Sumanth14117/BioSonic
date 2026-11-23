# BioSonic M5 — YAMNet transfer-learning experiment

## BirdCLEF dataset attribution and licensing

This report contains aggregate results derived from BirdCLEF+ 2026 competition recordings and metadata. Attribute BirdCLEF+ 2026 and the original recording contributors; the competition dataset is subject to **CC BY-NC-SA 4.0** and the competition rules. BioSonic does not claim ownership of the source data or dataset-derived material. This dataset license is separate from the Google YAMNet model listing's Apache-2.0 terms, AudioSet's dataset and ontology terms, and the root Apache-2.0 license for original BioSonic source code.

## Objective and scope

This is an eight-class interim recording-level primary-label classification experiment using the 2,177 approved recordings present in the working set. Four intended classes (`swtman1`, `smbani`, `yecpar`, `spispi1`) remain unavailable, so M5 does not complete the 12-class objective. The six M2 representative samples are excluded from all experiment partitions and are used only for the isolated YAMNet smoke test. This is not soundscape multi-label detection, nor a direct estimate of animal population size.

## Model, source, and license

The frozen YAMNet backbone is Google's MobileNetV1 depthwise-separable AudioSet model. It contains **3,751,369 parameters**, emits 521 AudioSet scores and 1,024-dimensional embeddings per patch, and its original classifier outputs are not used as BioSonic species predictions. Official model/checkpoint source: [Google YAMNet model card](https://tfhub.dev/google/yamnet/1); official implementation: [TensorFlow Models YAMNet at pinned commit c14bf9ad91962cf189f9f58db2132c06247fcd53](https://github.com/tensorflow/models/tree/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet); official weights: [https://storage.googleapis.com/audioset/yamnet.h5](https://storage.googleapis.com/audioset/yamnet.h5). The model card lists Apache-2.0. AudioSet's dataset terms are CC BY 4.0 and its ontology terms CC BY-SA 4.0; those are distinct from the model's license. Provenance, checksums, and retained license text are in `models/m5_yamnet/PROVENANCE.md` and `models/m5_yamnet/source_provenance.json`.

## Processing and embedding aggregation

Source OGG/Vorbis files were read unchanged using SoundFile as mono PCM16, then converted to float32 by dividing by 32768, matching the pinned official YAMNet `inference.py` example. Audio was resampled in memory to 16 kHz using resampy 0.4.3 (`kaiser_best`); output length follows resampy's floor convention. We did not clip finite post-resampling filter overshoot: the official reference inference path passes the resampled waveform directly to YAMNet. Its peak and count of samples outside [-1, 1] are recorded per file as a diagnostic. This preserves upstream behavior, but the resulting inputs do not always meet the model card's stated normalized range: the recorded maximum absolute value was 1.6589 for training and 1.6967 for holdout, with 32,783 and 9,855 samples outside [-1, 1], respectively. The models were run on those values without clipping; this is a material input-range limitation and should be addressed only in a separately designed future experiment with a new validation protocol. We did not use M4's 128-band cache. For every recording, the arithmetic mean of all 1,024-dimensional patch embeddings is the one recording-level vector; patches are not treated as independent training examples. Patch counts and source hashes are persisted in the M5 extraction manifests.

An M2 representative recording outside the approved experiment subset was used for the smoke test. It produced 25 patches, score shape [25, 521], embedding shape [25, 1024], and log-Mel shape [1248, 64]. Repeated CPU outputs matched within 1e-5 relative / 1e-6 absolute tolerance.

## Dataset, partitions, and leakage prevention

The authoritative split remains unchanged: **1,748 training recordings** and **429 final holdout recordings**. Classifier development used only the 1,748 training recordings with five-fold `StratifiedGroupKFold`, seed 42, stratification by `primary_label × collection`, and the existing author/content `group_id`. All five folds retained all eight classes, all 16 label×collection strata, and had zero group overlap. Unknown-author recordings remain in the original training partition and share the established unknown-author grouping. The existing final holdout was not read for embedding extraction, classifier fitting, CV, scaler fitting, hyperparameter choice, or model selection; it was first accessed after the classifier freeze marker was written.

## Classifier configurations and cross-validation

Macro-F1 was the primary selection metric. Logistic Regression and RBF SVM use a scikit-learn Pipeline with `StandardScaler` fit inside each training fold and `class_weight='balanced'`. Random Forest uses no scaler, `class_weight='balanced_subsample'`, four workers, and seed 42. The tested grids were LR C ∈ {0.1, 1, 10}, SVM C ∈ {0.5, 1, 5} with `gamma='scale'`, and RF 200/300 trees, max depth None/24, and leaf size 1/2. Candidate selection and final fit were restricted to training data.

Best configuration per model (mean and sample standard deviation across five author-grouped folds):

| model               | candidate_parameters                                                                 | mean_macro_f1 | std_macro_f1 | mean_accuracy | mean_precision_macro | mean_recall_macro | mean_weighted_f1 | mean_roc_auc_ovr_macro |
| ------------------- | ------------------------------------------------------------------------------------ | ------------- | ------------ | ------------- | -------------------- | ----------------- | ---------------- | ---------------------- |
| logistic_regression | {"model__C": 0.1}                                                                    | 0.5972        | 0.0517       | 0.5996        | 0.6087               | 0.6026            | 0.5959           | 0.8763                 |
| random_forest       | {"model__max_depth": null, "model__min_samples_leaf": 2, "model__n_estimators": 300} | 0.5303        | 0.0420       | 0.5423        | 0.5601               | 0.5347            | 0.5326           | 0.8648                 |
| svm                 | {"model__C": 5.0, "model__gamma": "scale"}                                           | 0.5661        | 0.0492       | 0.5679        | 0.5817               | 0.5721            | 0.5661           | 0.8641                 |

The selected frozen classifier is **logistic_regression** with parameters `{"model__C": 0.1}`. Its training-CV macro-F1 was 0.5972 ± 0.0517. Fold-level metrics and all candidate configurations are in `classifier_cv_folds.csv` and `classifier_cv_candidates.csv`.

## Final holdout evaluation

Only after the classifier artifact and freeze record were finalized, the pipeline processed the existing 429 recordings once. Results are recording-level:

- Accuracy: **0.6364**
- Macro precision: **0.6359**
- Macro recall: **0.6279**
- Macro F1: **0.6268**
- Weighted F1: **0.6356**
- OVR macro ROC-AUC: **0.8966**

Per-class scores:

| Class   | precision | recall | f1     | support |
| ------- | --------- | ------ | ------ | ------- |
| compot1 | 0.7879    | 0.7027 | 0.7429 | 74      |
| rufhor2 | 0.7424    | 0.7656 | 0.7538 | 64      |
| saytan1 | 0.6667    | 0.8182 | 0.7347 | 44      |
| crbthr1 | 0.5075    | 0.6182 | 0.5574 | 55      |
| rutjac1 | 0.4464    | 0.4630 | 0.4545 | 54      |
| rufnig1 | 0.6667    | 0.4390 | 0.5294 | 41      |
| chobla1 | 0.6444    | 0.5918 | 0.6170 | 49      |
| burowl  | 0.6250    | 0.6250 | 0.6250 | 48      |

The confusion matrix and recording-level predictions are saved separately in this M5 report directory. The shared holdout had already been used for prior M3 and M4 benchmark reporting; this M5 run follows the requested frozen final-evaluation boundary, but the comparison is not an independent external test.

## M3/M4/M5 comparison

| model                           | milestone | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | ovr_macro_roc_auc | selection_evidence                                                |
| ------------------------------- | --------- | -------- | --------------- | ------------ | -------- | ----------- | ----------------- | ----------------------------------------------------------------- |
| M3 svm                          | M3        | 0.6573   | 0.6679          | 0.6400       | 0.6431   | 0.6548      |                   | M3 training-only group CV                                         |
| M3 logistic_regression          | M3        | 0.6830   | 0.6857          | 0.6773       | 0.6769   | 0.6853      | 0.9244            | M3 training-only group CV                                         |
| M3 random_forest                | M3        | 0.6434   | 0.6507          | 0.6260       | 0.6315   | 0.6394      | 0.9061            | M3 training-only group CV                                         |
| M4 custom CNN                   | M4        | 0.8788   | 0.8913          | 0.8692       | 0.8726   | 0.8768      | 0.9851            | M4 internal validation; same previously evaluated M3 holdout      |
| M5 YAMNet + logistic_regression | M5        | 0.6364   | 0.6359          | 0.6279       | 0.6268   | 0.6356      | 0.8966            | Five-fold author/content-grouped CV on M3 training partition only |

On this shared holdout, M5 macro-F1 is below both M3 Logistic Regression and the M4 custom CNN results. This descriptive comparison uses the same historical holdout that had already been used in M3/M4 reporting, so it is not an independent external test. The M3/M4 artifacts were read for comparison and were not recalculated or overwritten.

## Computational performance and hardware

The isolated environment used TensorFlow 2.21.0 and tf-keras 2.21.0 on Python 3.12.10. Native Windows TensorFlow detected no GPU, so YAMNet embedding extraction ran on CPU; the PyTorch/CUDA environment and RTX 4060 were not used. The checkpoint is 15,296,092 bytes; the selected classifier is 55,838 bytes. `computational_performance.json` reports summed model inference time, the resumed training-aggregate reconstruction time, actual holdout extraction wall time, CV/refit time, and classifier prediction time. Because training extraction was resumed after an interrupted run, full end-to-end wall time across all extraction attempts was not captured. The 8 GB GPU is not a bottleneck for this CPU YAMNet run; native-Windows TensorFlow GPU support remains unavailable.

## Tests, audit, and reproducibility

The package test suite exercises input validation, resampling, patch aggregation, finite values, duplicate IDs, class alignment, train-only grouped folds, scaling inside pipelines, frozen holdout gating, output mapping, and artifact integrity. The final audit records 2,177 approved source mappings, the 1,748/429 partition boundary, source-hash/size/decode checks, zero training/holdout author/content overlap, and the excluded M2 samples.

Reproduce with the isolated environment (PowerShell):

```powershell
.\.venv-yamnet\Scripts\Activate.ps1
$env:PYTHONPATH = 'src'
python scripts/run_m5_yamnet.py --stage smoke
python scripts/run_m5_yamnet.py --stage extract-train
python scripts/run_m5_yamnet.py --stage train-freeze
python scripts/run_m5_yamnet.py --stage evaluate-holdout
python scripts/run_m5_yamnet.py --stage audit-report
```

The first three stages do not parse holdout labels or audio. `evaluate-holdout` refuses to run unless the selected classifier matches the M5 freeze manifest. The evaluation artifact prevents a second holdout evaluation. Do not retune based on final holdout results.

## Limitations and conclusion

This experiment covers only the eight species represented in the currently available approved subset. Author grouping does not guarantee geographic independence; unknown-author generalization is not measured separately. Recording-level primary labels are weak labels for local acoustic events. CPU inference timing is machine-specific. The same historical 429-recording holdout supports M3/M4/M5 comparison but repeated reuse limits independence. Acoustic detections and activity indices must not be interpreted as direct animal population counts or trends without a separate ecological sampling design.
