# BioSonic M2 — Controlled preprocessing experiment

## Method

All variants were created in memory from one decoded float64 waveform per recording. The source was never mutated, and no transformed audio was written. All variants retain the original 32 kHz sample rate, channel count, and sample count. No resampling, channel conversion, clipping, or training was performed.

- A: original decoded waveform `x`.
- B: mean removal, `x - mean(x)`.
- C: peak normalization, `x × (0.9 / max(abs(x)))`.
- D: RMS normalization, `x × (0.05 / RMS(x))`.
- E: mean removal followed by RMS normalization, `(x - mean(x)) × (0.05 / RMS(x - mean(x)))`.

All denominators were nonzero; no numerical fallback was needed. The CSV contains metrics for every recording/variant pair, including any percentage outside [-1, 1]. No out-of-range samples were clipped or repaired.

For spectral comparisons, every variant used the same NumPy Hann-windowed, unpadded STFT and the same librosa Slaney Mel filter bank (128 bands, `htk=False`, `norm="slaney"`). Mel power is `filterbank @ abs(STFT)^2`. For each recording, one shared dB reference is the largest Mel-power value among its five variants; values are shown as `10 log10(power / shared_reference)` with an 80 dB floor. The reference is shared within a recording, not across recordings.

## Key quantitative findings

Peak normalization sets maximum absolute amplitude to 0.9. RMS normalization sets RMS to 0.05. Mean removal makes the mean approximately zero, subject only to floating-point precision. See `preprocessing_comparison.csv` for all 30 rows and every requested metric.

Bare-faced Curassow original mean is 0.061744, RMS 0.091225, and standard deviation 0.067154. Mean removal shifts the mean to -2.381e-19; this is a substantial change because the original contains a persistent positive offset. Earlier five-second segment means were all positive (0.0520–0.0770), and the incomplete tail mean was 0.062476, so the offset was present throughout rather than limited to one segment.

Peak and RMS normalization use different scale factors because they target different properties. Peak normalization depends only on the single largest absolute sample; RMS normalization depends on average signal energy across the complete recording. The factors for each file are in the CSV.

## Interpretation

- Mean removal subtracts a constant from the whole recording. Its largest spectral effect is expected near DC and at the lowest Mel bands; higher-frequency call bands and their timing should remain broadly similar. The Curassow is the clearest example because of its strong positive mean.
- Peak normalization and RMS normalization multiply the waveform by a constant. This scales the entire Mel power spectrum by the square of that factor, changing absolute power while generally preserving relative spectral shapes and event timing. The shared dB reference makes the level shift visible rather than hiding it by per-variant normalization.
- Mean removal plus RMS normalization changes both the baseline and the overall scale. It can therefore alter low-frequency energy and shift remaining Mel energy levels.
- RMS normalization can amplify quiet background noise when a recording has low overall RMS; peak normalization can also raise quieter material when the peak is unusually small. The transformation is global per recording and cannot distinguish wildlife from noise.
- Values outside [-1, 1] are reported in the CSV. They are not automatically clipped. A normalized waveform exceeding that range is a diagnostic issue, not a reason to silently alter it.

## Limitations and validation

These six examples are a small, selected set, not a representative validation sample. They cannot establish whether mean removal or either normalization improves species classification, soundscape detection, or robustness. They also do not quantify performance across locations, recordists, or rare classes.

Choose preprocessing using recording-level validation: split by original recording (and preferably by site/source where appropriate), keep all windows from a recording in one split, and compare candidate pipelines using the same model, split, and evaluation metric. This avoids leakage from near-identical segments and measures whether a transformation helps on recordings the model did not see.

## Integrity

SHA-256 hashes matched before and after analysis for all six originals. Sample counts, sample rates, and durations were unchanged. Only the requested CSV, report, and comparison figures were written.
