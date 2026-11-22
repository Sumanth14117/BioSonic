# BioSonic — AI-Powered Wildlife Monitoring & Biodiversity Intelligence System

BioSonic is a research project exploring how machine learning can help monitor wildlife acoustically and support biodiversity analysis. The initial focus is bird audio classification using the BirdCLEF+ 2026 competition data. Longer-term work will examine wildlife detection in environmental soundscapes and summarize acoustic activity over time.

## Goals

- Build a reproducible pipeline for validating and analyzing wildlife recordings.
- Develop an initial bird recording classifier using species labels and documented validation splits.
- Explore soundscape detection, where multiple species may occur in the same time interval.
- Investigate how acoustic detections can support biodiversity intelligence while communicating uncertainty and limits clearly.

## Current status

### Completed

- **M1 — Dataset understanding:** reviewed the BirdCLEF+ 2026 training metadata, taxonomy, inventory, soundscape labels, and submission schema.
- **M2 — Audio exploration and preprocessing research:** completed waveform, FFT, STFT, Mel spectrogram, MFCC, segmentation, quality, and controlled preprocessing experiments on six representative recordings. A provisional preprocessing specification is documented. No preprocessing transformation has been adopted as a model default.

### In progress

- **M3 — Traditional machine learning:** the initial 12-species bird subset and a leakage-aware validation strategy have been reviewed. The approved audio acquisition is incomplete: **2,177 of 3,373 recordings are verified locally**. Dataset assembly and split implementation remain in progress.
- The first Logistic Regression, Random Forest, and SVM experiments have not yet produced model results. **No final model performance results are available.**

### Planned

- Complete audio verification and freeze the recording-level train/validation split.
- Extract documented features and compare traditional ML baselines.
- Train and evaluate a custom CNN and investigate transfer learning.
- Develop a separate multi-label soundscape detection task.
- Explore biodiversity summaries and a future service/dashboard interface.

## Dataset and attribution

The initial experiments use the **BirdCLEF+ 2026** competition dataset and associated metadata. The project data includes recordings from the collections identified in the competition metadata, including Xeno-canto (XC) and iNaturalist (iNat).

Dataset attribution: **BirdCLEF+ 2026 competition and the recording contributors identified by its metadata.** The applicable competition dataset terms specify **CC BY-NC-SA 4.0**. Attribute the competition and original contributors when redistributing permitted dataset-derived material, and follow the license and competition rules for non-commercial use and sharing adaptations. Dataset files are excluded from this Git repository. This dataset license applies to the dataset as specified by its terms; it does **not** automatically license BioSonic source code or other project materials. A separate license for BioSonic has not been selected.

## Planned audio and modeling workflow

```text
BirdCLEF metadata and approved recordings
        ↓
File, format, sample-rate, and decode validation
        ↓
Recording-level split and leakage checks
        ↓
Audio features and documented preprocessing experiments
        ↓
Traditional ML baselines → CNN / transfer-learning experiments
        ↓
Separate soundscape multi-label detection
        ↓
Biodiversity activity summaries → future API and dashboard
```

The first learning task is **recording-level, single-label classification** using `primary_label`. Soundscape detection is a distinct future task: it uses time intervals and may require multiple simultaneous species labels. Its labels and evaluation must not be mixed with the initial single-label experiment.

The exploratory feature work includes five-second, non-overlapping segmentation and power Mel spectrograms at 32 kHz, with a 1,024-point FFT, 256-sample hop, 128 Slaney-normalized Mel bands, and no frame padding. These are documented exploratory settings, not a claim that the final model configuration has been chosen.

## Architecture and repository layout

| Location | Purpose |
|---|---|
| `data/` | Local metadata, downloaded recordings, and future derived data. Ignored by Git. |
| `scripts/` | Project utilities, including the approved-audio downloader. |
| `src/` | Planned reusable application and ML components. |
| `notebooks/` | Planned exploratory notebooks. |
| `reports/` | Dataset analyses, audio experiments, figures, and progress artifacts. |
| `docs/` | Project specifications and technical documentation. |
| `tests/` | Mock-based regression tests and future project tests. |

The future application architecture is expected to separate data ingestion and validation, feature/model code, soundscape inference, biodiversity summaries, a **FastAPI** service, and a **React + TypeScript** interface. The API and frontend are planned components, not completed features.

## Milestone roadmap

| Milestone | Focus | Status |
|---|---|---|
| M1 | Research, environment setup, and dataset understanding | Complete |
| M2 | Audio preprocessing and exploratory analysis | Complete |
| M3 | Traditional machine learning | In progress |
| M4 | Custom CNN development and evaluation | Planned |
| M5 | Transfer learning and model comparison | Planned |
| M6 | Multi-species soundscape detection | Planned |
| M7 | Explainability and biodiversity analytics | Planned |
| M8 | FastAPI backend and React/TypeScript frontend | Planned |
| M9 | Testing, deployment, documentation, and portfolio release | Planned |

## Technology direction

- **Data and audio:** Python, Librosa, NumPy, Pandas, Matplotlib.
- **Traditional ML:** Scikit-learn (Logistic Regression, Random Forest, and SVM).
- **Deep learning:** PyTorch (custom CNN and transfer learning).
- **Future application:** FastAPI, React, and TypeScript.

Technology choices beyond the completed exploratory work remain subject to implementation and validation; no deployment or performance claims are made here.

## Hardware-aware development

Development uses an **NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM**. Experiments will be sized for that memory budget: begin with manageable subsets and model batches, monitor GPU memory, and use the GPU where it benefits PyTorch workloads. No speed benchmark or training-capacity claim is implied.

## Evaluation and ecological interpretation

Splits are designed around original recording identity so derived windows from one recording cannot cross train and validation partitions. Learned preprocessing must be fitted on training data only. Model comparisons and error analysis will be reported after experiments are run; no accuracy or benchmark is claimed yet.

Acoustic detections and activity summaries can help describe when and where sounds are recorded. They are **not equivalent to direct estimates of wildlife population size or abundance**. Any ecological interpretation will need appropriate validation, coverage, detectability, and uncertainty analysis.

## Data handling

`data/` and virtual environments are excluded by `.gitignore`. Do not commit competition audio, credentials, or local environment files. Reports and selected generated figures are currently eligible for tracking; review dataset-derived manifests and stale reports before including them in a public commit.
