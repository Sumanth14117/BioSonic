# BioSonic M2 — Audio quality and preprocessing audit

## Scope and method

This read-only audit used the six existing OGG recordings at their original 32 kHz sample rate. Whole-record statistics use every decoded mono sample. Complete 5-second segment statistics use the half-open sample ranges in `reports/segmentation/segment_manifest.csv`. Short-time RMS uses complete 3,200-sample (100 ms) frames with 1,600-sample (50 ms) hops; any unframed tail is reported in `audio_quality_summary.csv` and still contributes to whole-record statistics. No preprocessing was applied.

## Findings

- The six recordings are mono at 32,000 Hz. Whole-record mean amplitude ranges from -0.000071 to 0.061744; recording RMS ranges from 0.007282 to 0.091225.
- Bare-faced Curassow (`bafcur1/XC1018568.ogg`) has whole-record mean 0.061744, median 0.049550, and RMS 0.091225. Its 11 complete five-second segment means range from 0.052004 to 0.077030 (range 0.025026). Segment means: 0.077030, 0.066778, 0.052004, 0.067441, 0.053007, 0.067672, 0.053314, 0.068530, 0.052977, 0.067011, 0.052739. Compare these to the whole-record mean in the segment plot to judge whether the offset is steady or changes over time.
- `|amplitude| >= 0.99`: 0 samples across all six files (0.000000% of samples).
- `|amplitude| >= 0.999`: 0 samples across all six files (0.000000% of samples).
- `|amplitude| >= 1.0`: 0 samples across all six files (0.000000% of samples).
- Clipping thresholds are diagnostics only. Values near or above full scale may result from encoding/decoding behavior or signal peaks; sample values alone cannot prove hard clipping, which is better assessed from waveform shape and source encoding.
- Exploratory 100 ms frame RMS distributions and percentages below 0.001, 0.005, and 0.01 are in `audio_quality_summary.csv`. These cutoffs are not universal silence definitions.
- Amplitude percentiles and positive/negative/zero sample proportions are in `audio_quality_summary.csv`.

## Preprocessing recommendations (not applied)

| Operation | Recommendation | Reasoning and risk |
|---|---|---|
| DC offset correction | Do not apply globally yet. Compare raw input against per-recording mean removal in a controlled validation later. | A nonzero whole-record mean can indicate offset, but if it varies by time, a single correction may not remove it consistently. Per-segment correction can erase real low-frequency biological energy. |
| Peak normalization | Avoid per-recording peak normalization as a default. | One transient can set the peak and make the rest too quiet; rescaling every recording can also remove meaningful relative level differences. |
| RMS normalization | Avoid automatic per-file RMS normalization initially. | It may amplify quiet background noise and flatten ecological distance/abundance cues. Consider only as a documented training-only comparison. |
| Silence handling | Preserve quiet sections and timestamps; do not trim automatically. | Low-energy periods may represent species absence or context and soundscape evaluation is time-aligned. The RMS thresholds here are exploratory only. |
| Clipping | Keep thresholds as diagnostics; inspect any flagged examples before deciding on handling. | Threshold counts alone cannot establish clipping. Clipping repair can alter calls and introduce artifacts. |
| Noise reduction | Do not apply by default. | Wildlife calls can resemble noise to generic denoisers; suppression may remove quiet calls or alter harmonics and temporal texture. |
| Resampling | No resampling is needed for these files; retain 32 kHz. | All six already use the target rate. Resampling adds computation and can alter high-frequency content. |
| Mono conversion | No conversion is needed for these six mono files. | If future recordings are stereo, define a consistent downmix policy and check for phase cancellation before averaging channels. |

These recommendations are provisional and based on six recordings only. Validate preprocessing choices on held-out recordings and soundscape segments before adopting them.


## Incomplete five-second tails

The whole-record statistics include all samples. Segment statistics above use the 29 complete manifest segments; the final tail for each recording is separately summarized here and in `audio_quality_summary.csv`, but is not treated as a complete five-second segment.

- `XC1006659.ogg`: 72,032 samples (2.251000 s), mean -0.000070, RMS 0.003187.
- `XC1034681.ogg`: 50,624 samples (1.582000 s), mean 0.000001, RMS 0.007231.
- `XC1034537.ogg`: 49,120 samples (1.535000 s), mean -0.000014, RMS 0.011424.
- `XC1018568.ogg`: 150,016 samples (4.688000 s), mean 0.062476, RMS 0.087172.
- `XC892927.ogg`: 71,936 samples (2.248000 s), mean 0.000000, RMS 0.000920.
- `iNat103300.ogg`: 70,331 samples (2.197844 s), mean -0.000038, RMS 0.032540.

For Bare-faced Curassow, the complete-segment means range from 0.052004 to 0.077030 (standard deviation 0.008531); all 11 are positive. Its final 4.688-second tail mean is 0.062476, close to the whole-record mean 0.061744. This supports a positive offset that persists through the recording, with moderate segment-to-segment variation rather than an offset confined to one section.
