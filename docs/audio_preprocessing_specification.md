# BioSonic Audio Preprocessing Specification

**Status:** Provisional M2 specification. This document records current evidence and defines a conservative baseline; it does not apply preprocessing or freeze a final model configuration.

## 1. Purpose and scope

Define how BioSonic loads, validates, segments, and represents BirdCLEF+ 2026 audio for development. Preserve competition audio as immutable source data. Any transformed waveform or feature array is derived in memory or written as an explicitly versioned feature artifact; it never replaces the source recording.

This specification covers two separate learning tasks: recording-level primary-label classification and soundscape multi-label detection. They use different labels, sample units, and evaluation assumptions and must not be merged into one experiment.

The evidence available so far includes the BirdCLEF metadata files, the soundscape label file, and six selected OGG recordings. The six-file results are useful for pipeline exploration, not a representative estimate of full-dataset behavior.

## 2. Supported input audio format

The current supported input is the competition's `.ogg` audio, decoded as Vorbis by SoundFile. The six selected files all opened and decoded successfully. Preserve each original path, filename, and byte content.

Other codecs or extensions are not assumed to be supported until a file can be decoded and its properties validated. Do not rename a file to imply a different format or transcode it silently.

## 3. Audio loading and validation

Use `soundfile` / libsndfile to decode audio. For the current analysis baseline, use float64 arrays where practical; a future model-facing float32 conversion is allowed only as an explicit in-memory dtype choice. Decoding is not a source-file conversion.

For every input, record and validate:

- Decoded sample rate, channel count, sample count, duration (`samples / sample_rate`), dtype, and source path.
- Successful full decode and finite sample values (`NaN`/`Inf` count must be zero).
- Non-empty signal and expected channel shape.
- Source SHA-256 before and after an experiment when integrity verification is required.

A validation failure should stop that file's processing and be recorded; do not silently skip, repair, or substitute data. Retain sample count as the authoritative duration basis.

## 4. Sample-rate policy

The six selected recordings are mono at **32,000 Hz**. Treat 32 kHz as the current canonical rate for these experiments. Do not resample a source that already meets the target rate.

If a future source has a different rate, do not silently pass it into a 32 kHz feature pipeline. First decide whether to reject it or resample it for a specific experiment. If resampling is approved, record the original and target rates, resampling library/method, filter settings, and output dimensions. Keep the source file unchanged.

## 5. Channel policy

All six selected recordings have one channel. No mono conversion is needed for them.

Do not automatically convert multichannel files. For a task that requires mono, define the method before processing (for example, arithmetic channel averaging after confirming aligned channels), retain the channel count and downmix method in metadata, and check for phase cancellation. Preserve the source channels.

## 6. DC offset policy

Do not automatically remove DC offset. The quality audit found near-zero recording means for five recordings and a positive mean of **0.061744** for the Bare-faced Curassow. Its 11 complete five-second segment means were all positive (0.052004–0.077030), and the final 4.688-second tail mean was 0.062476. This supports a persistent offset in that clip, but does not establish whether the offset is recording bias or useful low-frequency signal.

Mean removal (`x - mean(x)`) remains a validation ablation. Compare it with unmodified audio using identical recording-level splits. Do not subtract the mean separately in every segment by default; that can remove real low-frequency content and make preprocessing depend on window boundaries.

## 7. Amplitude normalization policy

Do not automatically apply per-recording peak or RMS normalization. The controlled experiment showed that scale factors differ by recording: peak normalization to 0.9 ranged from about 1.220 to 15.038; RMS normalization to 0.05 ranged from about 0.548 to 6.866. Normalization can alter meaningful relative levels and can amplify quiet background along with calls.

If normalization is evaluated, record its target, scale factor, scope (recording or segment), and whether mean removal precedes it. Compare it against the original waveform in a controlled validation. Do not clip normalized values silently.

## 8. Clipping detection policy

Keep clipping checks diagnostic. For exploratory audits, count samples with `abs(amplitude) >= 0.99`, `>= 0.999`, and `>= 1.0`, and report counts and percentages alongside extrema. None of the six selected originals crossed these thresholds.

These checks do not prove whether hard clipping occurred earlier in the recording chain; codec behavior, gain, and the signal itself affect sample values. Inspect flagged waveform regions and source information before deciding on handling. Do not automatically clip, repair, or discard flagged audio.

## 9. Silence and low-energy handling

Do not discard or trim frames using arbitrary RMS thresholds. The quality audit calculated exploratory 100 ms frame RMS at thresholds 0.001, 0.005, and 0.01, but these are not universal definitions of silence.

Quiet periods can represent species absence, ambience, or low-level calls and matter for soundscape detection. If a later task uses silence trimming or frame filtering, define its threshold and purpose, preserve original timestamps, and evaluate the effect on missed detections and false positives.

## 10. Noise reduction policy

Do not apply generic noise reduction by default. Wildlife calls may resemble noise to a denoiser; suppression can remove quiet calls, alter harmonics, or smear short events. Evaluate any denoising method as a separate, versioned experiment against the unmodified source, with recording-level validation and checks for call preservation.

## 11. Fixed-length segmentation policy

Use **five-second, non-overlapping windows** as the initial fixed-length training baseline: at 32 kHz, each window contains **160,000 samples**. This aligns with the five-second intervals in `train_soundscapes_labels.csv` and the sample-submission row pattern. It is a useful starting window, not a final claim that five seconds is optimal.

The existing exploratory manifest uses half-open sample intervals `[start, end)`, assigns segment indices from zero, and preserves the original filename and primary label. The six selected recordings produced 29 complete windows. Do not save derived audio segments unless a later task explicitly requests them.

Keep the original recording identity, source path, label, sample offsets, and segment index with every derived segment or feature. A recording-level primary label applied to every window is a weak label: the target species may not be audible in every segment, and other species may be present.

## 12. Incomplete-tail handling

Do not silently discard incomplete final audio. The initial complete-window manifest excludes tails from its segment rows, but keeps the original recording intact and reports tail sample count/duration separately. The quality audit also retained tail samples in whole-record statistics.

For future training, compare explicit policies such as retaining complete windows only, padding a final window with a documented mask/pad value, or handling the shorter tail separately. Ensure padded values cannot be mistaken for real quiet audio. For soundscape evaluation, follow the required interval and row-ID convention exactly.

## 13. Mel spectrogram configuration

The current exploratory power-Mel baseline is:

| Parameter | Value |
|---|---:|
| Sample rate | 32,000 Hz (original; no resampling) |
| `n_fft` | 1,024 |
| `win_length` | 1,024 samples |
| `hop_length` | 256 samples (8 ms at 32 kHz) |
| Window | Hann (`numpy.hanning` in the exploratory implementation) |
| Mel bands | 128 |
| `fmin` / `fmax` | 0 / 16,000 Hz |
| Mel scale | Slaney (`htk=False`) |
| Filter normalization | Slaney (`norm="slaney"`) |
| Representation | Power (`abs(STFT) ** 2`, then Mel filter-bank projection) |
| Framing | Complete frames only; no centering or boundary padding |

The STFT has 513 nonnegative linear-frequency bins. Librosa constructs the 128 × 513 Slaney filter bank; NumPy applies it to each frame's power spectrum. With no padding, frame count is `1 + floor((sample_count - 1024) / 256)` for inputs at least 1,024 samples long.

This is an **exploratory configuration**, not a final model specification. The plots used log-power/dB for visualization. Previous plots used a per-recording maximum reference; the controlled comparison used a shared reference across five variants of each recording. Neither visualization convention should be silently treated as the model's final feature scaling. Select any log reference, floor, or power compression explicitly and validate it.

## 14. Feature normalization

Do not normalize every recording or spectrogram independently by default. Per-recording normalization can erase absolute-level information and make the same input depend on its recording context.

If a model requires standardized features, fit the scaler or normalization statistics on the training split only, then apply those fixed values to validation and test data. Record the transform, axes, fitted statistics, and feature version. Keep waveform amplitude normalization separate from feature-domain normalization. Do not use validation/test data to fit normalization parameters.

## 15. Data leakage prevention

Split by **original recording before generating segments or transformed variants**. All windows, augmentations, and preprocessing variants from one source recording must remain in the same split. For soundscapes, group by soundscape filename; never put intervals from one soundscape in both train and validation.

Where metadata permits, test stricter grouping by site, date, recorder, or source collection to assess generalization. Fit scalers, thresholds, and any learned preprocessing only on training data. Keep the split manifest and random seed with the experiment.

## 16. Reproducibility and experiment tracking

For each run, record:

- Source filename/path, primary label where applicable, source SHA-256, sample rate, channel count, sample count, and duration.
- Segment policy and sample offsets, including tail handling.
- All preprocessing and feature parameters, order of operations, dtype, and library versions.
- Train/validation recording IDs, split method, random seed, evaluation metric, and code revision.
- Output artifact names and checksums; distinguish exploratory plots from model inputs.

The inspected environment used Python 3.14.7, SoundFile 0.14.0, NumPy 2.5.5, and librosa 1.0.1. No project dependency lockfile was present during M1 inspection; capture or establish an explicit environment specification before relying on exact numerical reproducibility. Keep original data immutable.

## 17. Known limitations

- Only six deliberately selected recordings have been explored; results do not establish full-corpus quality or model performance.
- The audio inventory lists 35,525 `train_audio/` entries while `train.csv` has 35,549 rows, a difference of 24 that still needs filename-level reconciliation.
- `train.csv` has 35,549 rows and 206 unique primary labels; taxonomy has 234 labels. All training primary labels map to taxonomy, but 28 taxonomy labels lack individual-recording primary-label examples.
- The soundscape annotation CSV has 1,478 raw rows, but every exact `(filename, start, end, primary_label)` row appears twice; this corresponds to 739 distinct interval annotations across 66 files. Deduplicate exact annotation duplicates for analysis/training so they do not receive accidental double weight.
- Soundscape labels are semicolon-separated multi-label values. The annotations contain 75 unique labels, all mapped to taxonomy; 28 of those labels are absent from individual-recording primary labels in `train.csv`.
- The six-file quality audit found a persistent positive mean offset in one recording and no samples crossing the tested clipping thresholds. Neither finding generalizes to the full dataset or proves source-chain clipping status.
- Low-energy thresholds, normalization, DC correction, window length, and Mel scaling have not been validated for classification performance.

## 18. Future experiments

Compare candidate preprocessing choices with recording-level splits and identical evaluation conditions:

- Raw waveform versus recording-level mean removal, including low-frequency effects.
- No waveform normalization versus peak and RMS normalization; measure background-noise amplification and level sensitivity.
- Complete-only windows versus padding or a documented partial-tail policy.
- Five-second windows versus alternate durations and overlap, with all segments grouped by source recording.
- Raw power-Mel versus explicit log-power reference/floor and train-only feature standardization.
- Mono handling for future multichannel sources, including phase-cancellation checks.
- Denoising only if a specific method preserves quiet wildlife calls on held-out recordings.
- Resampling only if future inputs do not meet the required model sample rate.

### Separate learning tasks

**A. Recording-level primary-label classification**

This is the initial traditional-ML learning task. Use `train.csv` `primary_label` as the recording target and preserve `secondary_labels` as metadata rather than silently merging them into the target. If fixed windows are generated, assigning the recording label to every window can introduce label noise; retain recording identity and evaluate with recording-grouped splits.

**B. Soundscape multi-label detection**

This is a separate task over five-second soundscape intervals, with multiple simultaneous species labels. Parse semicolon-separated annotation values into a multi-hot target and use **234 output classes** aligned exactly to taxonomy and the sample-submission schema, including classes without individual-recording training examples. The soundscape annotation file has duplicate exact rows that must be handled explicitly. Use a multi-label metric appropriate to the competition requirements (the supplied competition research identifies macro ROC-AUC, excluding classes without positive evaluation examples). Do not apply recording-level single-label assumptions or combine its evaluation with Task A.

### M2 completion checklist

| Step | Work | Status |
|---:|---|---|
| 1 | Select representative audio sample and download/verify six recordings | Complete |
| 2 | Waveform analysis | Complete |
| 3 | FFT and frequency-domain analysis | Complete |
| 4 | STFT and linear spectrogram analysis | Complete |
| 5 | Mel spectrogram analysis | Complete |
| 6 | MFCC, delta, and delta-delta analysis | Complete |
| 7 | Five-second segmentation analysis | Complete |
| 8 | Audio quality and preprocessing audit | Complete |
| 9 | Controlled preprocessing comparison | Complete; no transform adopted |
| 10 | Finalize preprocessing specification | Complete with this document |

### Proposed M3 outline

1. Select a manageable dataset subset and verify its files against metadata.
2. Recheck metadata, taxonomy mappings, class counts, and label assumptions.
3. Create a recording-level train/validation split before segmentation or feature fitting.
4. Extract a documented baseline audio feature set.
5. Train a baseline Logistic Regression model.
6. Train a Random Forest model.
7. Train an SVM model.
8. Compare models using the same split and suitable metrics; document class coverage and limitations.
9. Perform error analysis by species, collection/source, and recording conditions.

M3 should begin only after the split and label policies are reviewed. No preprocessing or model configuration in this specification has been applied to the source audio.

## Dataset provenance and licensing

The exploratory audio and figures described here were derived from BirdCLEF+ 2026 competition recordings. Attribute BirdCLEF+ 2026 and original recording contributors; the dataset is subject to CC BY-NC-SA 4.0 and the competition rules. BioSonic does not claim ownership of these recordings or dataset-derived material. The root Apache-2.0 license applies to original BioSonic source code only and does not replace dataset, model, checkpoint, or vendored-code terms.
