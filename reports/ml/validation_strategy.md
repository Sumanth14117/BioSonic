# M3 Step 3 — Leakage-Safe Validation Strategy

**Scope:** Finalized design for the initial 12-species, single-label bird classification task. Split assignments and SHA-256 comparisons were checked in memory only; no split manifest was saved. No audio was downloaded or modified, and no models or preprocessing transformations were fitted. A read-only local-audio snapshot was hashed while the separate download was active; it must be repeated after download completion before creating the split manifest.

## Recommended strategy

Use a fixed **80% training / 20% validation** holdout from the previously evaluated five-fold `StratifiedGroupKFold` assignment (scikit-learn 1.9.1, `n_splits=5`, `shuffle=True`, `random_state=42`). Stratify on the joint key **`primary_label × collection`**. Group globally by normalized `author`; keep all literal `Unknown` author rows in one group. Use the second fold as validation only when the eligible recording set and deterministic checks below match. Each row is an original recording identified by its `filename`.

The second fold was recomputed twice in memory using the original `train.csv` row order, the 3,373 inventory-backed filenames, the two approved exclusions, `y = primary_label + '|' + collection`, and author strings normalized with Unicode NFKC, whitespace trimming, and case-folding. All 171 literal `Unknown` records were kept in one group. Both runs produced the same 676 validation filenames: 391 XC / 285 iNat, 46–73 records per species, and **zero author-group overlap**. No `Unknown` records are in this fold, so all remain in training. A SHA-256 fingerprint of the sorted validation filename list is `4ab26db69be7a71acff8be8d593fdb1be611fa433279854bcf6d7fb708410dc8`. The fingerprint is a reproducibility check, not a split manifest. If the final verified eligible set differs from these 3,373 rows, regenerate and verify the assignment on that exact set rather than reusing row indices.

## Selected-set counts and source balance

The metadata has 3,375 rows for these labels. The approved inventory-backed subset has 3,373 filenames after excluding the two inventory-missing files (`swtman1/XC299145.ogg` and `crbthr1/XC670199.ogg`). The table below uses the 3,373 intended recordings.

| `primary_label` | Common name | Recordings | XC | iNat | Known authors | Repeated author groups | Largest author group |
|---|---|---:|---:|---:|---:|---:|---:|
| `compot1` | Common Potoo | 368 | 185 | 183 | 182 | 51 | 32 |
| `swtman1` | Swallow-tailed Manakin | 354 | 228 | 126 | 132 | 69 | 16 |
| `rufhor2` | Rufous Hornero | 315 | 163 | 152 | 126 | 41 | 20 |
| `smbani` | Smooth-billed Ani | 305 | 230 | 75 | 157 | 58 | 18 |
| `saytan1` | Sayaca Tanager | 278 | 149 | 129 | 106 | 46 | 30 |
| `crbthr1` | Creamy-bellied Thrush | 268 | 156 | 112 | 115 | 40 | 24 |
| `rutjac1` | Rufous-tailed Jacamar | 265 | 228 | 37 | 99 | 45 | 40 |
| `rufnig1` | Rufous Nightjar | 256 | 112 | 144 | 89 | 35 | 49 |
| `chobla1` | Chopi Blackbird | 244 | 160 | 84 | 121 | 34 | 27 |
| `yecpar` | Yellow-chevroned Parakeet | 243 | 127 | 116 | 130 | 42 | 32 |
| `spispi1` | Spix's Spinetail | 239 | 118 | 121 | 122 | 40 | 28 |
| `burowl` | Burrowing Owl | 238 | 130 | 108 | 132 | 45 | 13 |
| **Total** | **12 species** | **3,373** | **1,986** | **1,387** | **774 distinct known author values across set** |  |  |

Counts are reasonably balanced by species (238–368 recordings) but source balance varies: for example, Rufous-tailed Jacamar is 228 XC / 37 iNat and Smooth-billed Ani is 230 / 75. Stratifying on the joint species/source key helps retain source representation. It does not remove source-related differences; report per-species results and XC/iNat diagnostics.

## Filename, author, and grouping findings

- In all of `train.csv`, all 35,549 `filename` values are unique. The 3,373 intended subset filenames are also unique. Selected `url` values are unique as well. This supports filename as the original-recording key, but uniqueness does not prove the audio contents are unique; compare SHA-256 hashes of available files before making a manifest and group byte-identical recordings together if any are found.
- After normalizing author text, the selected set contains 774 known author values and a literal `Unknown` value on 171 rows. There are 358 repeated author groups (including the unknown bucket), and 265 known/unknown groups span more than one species; 2,641 recordings are in multi-species author groups. The largest author group contains 240 recordings and spans all 12 selected species. These repetitions make an ordinary row-random split vulnerable to author/recording-condition leakage.
- Author counts per species range from 89 to 182 known values. The largest per-author contribution ranges from 13 to 49 recordings, or about 4.5%–19.1% of the species count. This is enough repetition to justify author grouping, while still leaving many groups for a holdout.
- `collection` is XC or iNat, not a unique recording identifier; it should be a stratification/diagnostic field, not the grouping key. Ten author values appear in both collections. Grouping author globally is conservative against the same named contributor appearing in both sources, though it may merge homonymous names.
- `train.csv` includes latitude and longitude for all 3,373 intended rows, with 2,133 distinct exact coordinate pairs and many repeated pairs. Coordinates may reveal site overlap, but the metadata has no explicit site, date, or recorder ID, and coordinate precision/meaning is not established. Do not use exact coordinates as the primary grouping key yet. Consider a separate spatial-block holdout after defining and documenting a defensible spatial resolution.
- Treat `Unknown` as one opaque author group for both the outer holdout and inner group-aware folds. For the selected second-fold holdout, all 171 such rows are in training. The validation score therefore does not measure performance on anonymous-author recordings as a separate validation population; `Unknown` is not evidence these recordings share a real contributor.

## Why group at author level, not only recording level

Filename is the required unit of isolation: it prevents the same original recording from appearing on both sides. But because filenames are unique, using filename alone as the group is equivalent to a row-level split and does **not** prevent recordings by the same contributor from crossing the boundary. Repeated author groups, including groups spanning all species, can carry recorder, microphone, environment, and upload-source characteristics into both partitions.

The recommended author-grouped split estimates performance on recordings from known contributors not represented in training. This is a stricter and useful generalization test, but it may be more pessimistic than deployment where known contributors recur. It can also alter collection proportions because many authors are source-specific. Since all `Unknown` rows remain in training, validation generalization to anonymous contributors is not measured. A future spatially blocked test can measure generalization to new locations.

## Split and feature-fitting procedure

1. After Step 2 completes, re-inspect `birdclef_file_inventory.csv` and the local audio tree. Form the eligible set by joining the approved filenames to `train.csv`; exclude a file only for a documented failure (missing, size mismatch, decode failure, or agreed quality criterion), retain an audit reason, and recompute class/source counts before splitting. This strategy review does not require waiting for the active downloader; do not interrupt or otherwise interfere with it. Complete this gate after it finishes and before creating the manifest.
2. Compute SHA-256 for each audio file already present and stable at the time of the pre-manifest audit. Report exact duplicate-hash groups; group duplicate-content filenames together or resolve them before splitting. Files absent at the snapshot cannot be assessed, and distinct filenames/URLs do not prove distinct content. Repeat the audit against the completed eligible set before writing the manifest.
3. Normalize author strings consistently (Unicode NFKC, trim whitespace, case-fold). Group equal known author values globally and all literal `Unknown` rows together. Preserve `filename` as the original recording/group identifier.
4. Reproduce the five-fold assignment using the exact eligible filename set, original row ordering, `y = primary_label + '|' + collection`, normalized author groups, `shuffle=True`, and seed 42. Verify the fold-2 filename fingerprint and all group-disjointness/balance checks. Only then adopt it as the fixed 80/20 holdout.
5. Save a parent-recording split assignment only after review. Create segments, crops, augmentations, or other variants only after this assignment; every derived example inherits its parent's split. Assert that no original `filename`, author group, or verified duplicate-content group crosses train/validation.
6. Tune features and hyperparameters using group-aware cross-validation only within the training partition. Fit scalers, imputers, feature selectors, normalization statistics, and every other learned transformation on each inner training fold only, then apply them unchanged to that fold's validation portion. For the final holdout evaluation, fit on the full training partition and transform validation without fitting on it.
7. Do not use the final holdout for feature selection, hyperparameter tuning, threshold selection, or repeated model choice. Report preselected model results on it once. If choices are changed after seeing those results, the holdout has become development data; use a new untouched evaluation set for a final claim.

Do not use competition test data or soundscape labels in this recording-level split. The target is the 12-class `primary_label`; `secondary_labels` remain metadata. No validation recordings are used to fit preprocessing.

## Validation checks before saving a split manifest

- Exactly one split assignment per eligible original filename; no duplicate or missing IDs.
- Every row has a target among the approved 12 `primary_label` values and maps to Aves in `taxonomy.csv`.
- No intersection of original filenames, normalized author groups (including the one Unknown bucket), or verified duplicate-content groups across train and validation.
- Confirm all segments/variants later inherit the parent recording's split; assert no segment-level crossing.
- Validation is approximately 20%; print per-class counts and per-class XC/iNat counts and proportions. Ensure all 12 labels are represented on both sides; if exclusions break this, revisit eligibility or use group-aware folds rather than moving individual recordings across groups.
- Record the split seed, scikit-learn version, group normalization rules, eligible-file list/hash, fold choice, validation filename fingerprint, and any exclusions.
- Confirm all learned preprocessing state was fit only on training data; confirm test data and validation labels were not used to fit it.

## Advantages and limitations

**Advantages:** protects original recording identity and known contributor groups; uses the full approved dataset without an arbitrary class cap; joint stratification retains a practical balance of species and source; the reproducible fold-2 assignment gives an approximately 20% holdout.

**Limitations:** author metadata may be inconsistent or homonymous; the 171 `Unknown` rows cannot be attributed to real contributors and all remain in training, so their generalization is not measured on the holdout. Author grouping may shift source proportions and reduce effective independent data; group stratification is approximate and cannot guarantee exact class/source quotas. Filename/URL uniqueness does not rule out duplicate audio. A point-in-time SHA-256 audit while download was active found 2,004 stable local OGG files, all inventory-size matches, with no exact duplicate hashes; this does not cover files downloaded later or files not yet present. Coordinates exist for all selected rows (2,133 exact pairs; 501 repeated pairs, maximum group size 31), but no explicit site/date/recorder key or validated coordinate resolution exists. Use coordinate/site grouping only in a separately designed future spatial-generalization experiment. Results from a single holdout still have sampling uncertainty.

**Recommendation:** adopt the fixed 80/20 author-grouped holdout from fold 2, stratified by `primary_label × collection`, seed 42, only if the exact eligible set reproduces the recorded fingerprint and verification checks. All `Unknown` author records remain in training. Use group-aware CV only inside training and never tune against the final holdout. Before creating the split manifest, repeat inventory/local availability checks and SHA-256 duplicate checks after the active download completes; coordinate grouping remains a separate future experiment.
