# Eight-class interim validation strategy

This split applies only to the **2,177 verified, locally available approved recordings**. It includes the eight classes present in that working set: compot1, rufhor2, saytan1, crbthr1, rutjac1, rufnig1, chobla1, burowl. It is not a 12-class result; `swtman1`, `smbani`, `yecpar`, and `spispi1` are unavailable and excluded from this experiment. The six M2 sample recordings outside the approved manifest are excluded.

## Split

- Fresh deterministic grouped holdout from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; selected candidate fold 5 as the feasible fold nearest 20% validation.
- Training: 1,748; validation: 429 (19.71%).
- Stratification: `primary_label × collection` (XC/iNat).
- Group key: connected components of normalized author identity and exact SHA-256 duplicate identity. All literal `Unknown` authors form one group and remain in training (109 recordings).
- All eight labels and all observed label×collection strata occur in both partitions. No author/content group crosses partitions.

## Leakage controls and limits

Split source recordings before segmentation. Any derived segment, feature vector, or transformation inherits its source partition. Learned preprocessing and feature selection are fit only on training folds. Group-aware cross-validation and model selection use training data only; this validation holdout remains untouched until final evaluation. Author grouping reduces same-contributor leakage but does not guarantee geographic independence or prevent near-duplicate recordings by different authors. The 109 Unknown-author recordings are training-only, limiting evaluation of unseen/unknown contributors. Spatial grouping is a separate future analysis.

The split is valid for the current eight-class subset only. It must be regenerated if the working dataset changes; the historic 676-recording holdout from the full intended dataset is not reused.
