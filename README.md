# BioSonic — AI-Powered Wildlife Monitoring & Biodiversity Intelligence System

BioSonic explores how machine learning can help monitor wildlife acoustically and support biodiversity analysis. The current research focuses on recording-level bird classification with BirdCLEF+ 2026 data. Soundscape detection and biodiversity summaries remain separate future tasks.

## Current results

The current interim experiment covers **eight bird classes** using **2,177 verified recordings** from the approved 3,373-recording set. Acquisition is incomplete: the current manifest records 2,177 verified, 1 failed, and 1,195 not attempted. Four approved classes (`swtman1`, `smbani`, `yecpar`, `spispi1`) have no verified recordings in this working subset. The six M2 demonstration recordings outside the approved set are excluded. The 12-class objective is not complete.

| Experiment | CV/validation-selected model | Holdout accuracy | Holdout macro-F1 | Notes |
|---|---|---:|---:|---|
| M3 traditional ML | RBF SVM | 0.6573 | 0.6431 | Selected by training-CV macro-F1 (0.6575). Logistic Regression scored 0.6830 accuracy / 0.6769 macro-F1 on the same holdout, but was not the CV-selected model. |
| M4 custom CNN | Four-stage CNN | 0.8788 | 0.8726 | Internal validation macro-F1 was 0.7603; trained on the M3 training partition. |
| M5 transfer learning | Frozen YAMNet embeddings + Logistic Regression | 0.6364 | 0.6268 | Training-only grouped CV macro-F1 was 0.5972 ± 0.0517; TensorFlow inference ran on CPU. |

These are **interim eight-class results**, not independent external estimates. All three milestones report on the same 429-recording holdout, which was already used in M3 and M4. Reuse limits the independence of cross-model comparisons. See the [M3](reports/ml/m3_eight_class/m3_eight_class_experiment_report.md), [M4](reports/ml/m4_custom_cnn_experiment_report.md), and [M5](reports/ml/m5_yamnet/m5_yamnet_transfer_learning_report.md) reports for methods and per-class detail.

## Preprocessing and model inputs

The M2 audio analyses are exploratory. M3 and M4 use five-second mono windows and a 32 kHz, 128-band Slaney power-Mel representation; M4 additionally pads incomplete tails for its CNN input. M5 follows YAMNet's distinct waveform interface: PCM16 audio is converted to float32 and resampled in memory to 16 kHz, then processed by YAMNet's own frontend. The M4 Mel cache is not used as YAMNet input.

In M5, resampling produced finite values outside the nominal `[-1, 1]` waveform range (maximum absolute value 1.6967). The experiment did not clip those samples, matching its recorded inference path. This is a known input-range limitation, not a recommended normalization policy. A future correction requires a separately validated experiment.

## Milestones

| Milestone | Focus | Status |
|---|---|---|
| M1 | Research, environment setup, and dataset understanding | Complete |
| M2 | Audio preprocessing and exploratory analysis | Complete |
| M3 | Traditional machine learning | Complete for the eight-class interim experiment |
| M4 | Custom CNN development and evaluation | Complete for the eight-class interim experiment |
| M5 | Transfer learning and model comparison | Complete for the eight-class interim experiment |
| M6 | Model optimization and improvement | Complete as a training-only optimization study; no improvement over the existing best M3 CV result |
| M7 | Explainability and biodiversity analytics | Planned |
| M8 | FastAPI backend and React/TypeScript frontend | Planned |
| M9 | Testing, deployment, documentation, and portfolio release | Planned |

M6 tested 15 classifier configurations on existing frozen YAMNet embeddings using author/content-grouped CV on the 1,748-recording training partition. Balanced Logistic Regression (`C=0.1`) remained the best M6 candidate at macro-F1 0.5972 ± 0.0517, matching the prior M5 baseline; the M3 RBF SVM remains the strongest comparable training-CV result at 0.6575 ± 0.0397. The reused holdout was not accessed. See [the M6 optimization report](reports/ml/m6_optimization/m6_optimization_report.md). M7–M9 remain planned. Acoustic activity estimates must not be presented as direct wildlife population estimates.

## Environments and commands

Two Windows environments are documented separately because PyTorch/CUDA and native-Windows TensorFlow CPU use different Python versions. The files are exact `pip freeze` snapshots of the existing environments, including installed tooling; **fresh installation from these manifests has not been tested**.

Main ML environment: Python 3.14.7, PyTorch 2.15.0+cu132, torchaudio 2.11.1+cu132, torchvision 0.29.2+cu132, and CUDA 13.2. CUDA was verified with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM). Use the official PyTorch wheel index pinned in [requirements-main.txt](requirements-main.txt).

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch==2.15.0+cu132" "torchaudio==2.11.1+cu132" "torchvision==0.29.2+cu132" --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements-main.txt
```

YAMNet environment: Python 3.12.10, TensorFlow 2.21.0, and tf-keras 2.21.0. In this project, native-Windows TensorFlow has no GPU execution; YAMNet runs on CPU and does not use the RTX 4060.

```powershell
py -3.12 -m venv .venv-yamnet
& .\.venv-yamnet\Scripts\python.exe -m pip install -r requirements-yamnet.txt
```

The current test suites were verified after M6: 84 tests passed in `.venv` (M3, M4, downloader, M6) and 13 passed in `.venv-yamnet` (M5). Example commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_download_approved_audio.py tests\test_m3_pipeline.py tests\test_m4_cnn.py -v
.\.venv\Scripts\python.exe -m pytest tests\test_m6_optimization.py -v
& .\.venv-yamnet\Scripts\python.exe -m pytest tests\test_m5_yamnet.py -v
```

The workflows have explicit stages and write local caches/results. They require the excluded dataset and, for M5, the separately acquired YAMNet checkpoint. Consult each experiment report before running stages; holdout evaluation is a one-time final step and must not be used for development.

From the repository root, the workflow entry points accept these stages:

```powershell
.\.venv\Scripts\python.exe scripts\run_m3_pipeline.py --stage validate
.\.venv\Scripts\python.exe scripts\run_m3_pipeline.py --stage features
.\.venv\Scripts\python.exe scripts\run_m3_pipeline.py --stage train-evaluate
.\.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage prepare
.\.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage mels
.\.venv\Scripts\python.exe scripts\run_m4_cnn.py --stage train
& .\.venv-yamnet\Scripts\python.exe scripts\run_m5_yamnet.py --stage smoke
& .\.venv-yamnet\Scripts\python.exe scripts\run_m5_yamnet.py --stage extract-train
& .\.venv-yamnet\Scripts\python.exe scripts\run_m5_yamnet.py --stage train-freeze
.\.venv\Scripts\python.exe scripts\run_m6_optimization.py --run-id run_02
```

These are reproducibility references, not commands to rerun during review. They may regenerate caches, models, or reports. M5 `evaluate-holdout` is intentionally omitted from routine development instructions because the shared holdout has already been evaluated; do not run it for tuning.

## Project layout

| Path | Purpose |
|---|---|
| `src/biosonic_ml/` | M3 recording-level features, split validation, and classical models. |
| `src/biosonic_cnn/` | M4 Mel extraction, CNN, dataset, and training pipeline. |
| `src/biosonic_yamnet/` | M5 isolated YAMNet input, embedding, and classifier workflow. |
| `scripts/` | Downloader and M3–M5 workflow entry points. |
| `tests/` | Downloader and milestone regression tests. |
| `docs/` | Audio preprocessing specification and publication checklist. |
| `reports/` | Aggregate analyses and experiment reports; review generated outputs before publication. |
| `models/` | Local checkpoints, classifiers, and model provenance. Weights are ignored by Git. |
| `data/` | Local metadata, audio, and derived data. Ignored by Git. |

## Technology stack

The implemented research pipeline uses Python, Librosa, NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow/tf-keras for CPU YAMNet inference, and Matplotlib. **FastAPI** and a **React + TypeScript** frontend are planned for M8; they are not implemented.

## Dataset and model provenance

The dataset is from [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026). The competition data is subject to **CC BY-NC-SA 4.0** and the competition rules; see the [dataset terms](https://www.kaggle.com/competitions/birdclef-2026/data) and [license text](https://creativecommons.org/licenses/by-nc-sa/4.0/). When sharing dataset-derived reports or results, attribute BirdCLEF+ 2026 and the original recording contributors, identify and link the applicable license, and indicate changes where applicable. Review the NonCommercial and ShareAlike terms before redistribution. BioSonic does not claim ownership of the recordings or dataset-derived material.

The root [LICENSE](LICENSE) applies Apache License 2.0 to original BioSonic source code only. It does not automatically apply to BirdCLEF-derived reports or results, competition data, or third-party components, and it does not replace their separate terms. No copyright-holder name has been added to the generic license text because ownership notice details were not established. M5 uses Google's YAMNet; its model listing identifies Apache-2.0, while the direct checkpoint blob has no independently verified license file. The checkpoint's license attribution therefore follows the model listing and is not an independent verification. Vendored upstream inference source retains its separate Apache-2.0 notice; provenance is documented in [models/m5_yamnet/PROVENANCE.md](models/m5_yamnet/PROVENANCE.md). The YAMNet model/source terms are distinct from BirdCLEF data terms and the separate AudioSet dataset/ontology terms.

Competition audio, metadata downloads, model weights/checkpoints, and generated feature caches are not included in this repository. Do not infer permission to redistribute them from the source-code or model licenses.

## Interpretation limits

The task is recording-level single-label classification using `primary_label`; those labels may be weak for individual windows. Multi-species soundscape detection requires its own labels, outputs, and evaluation. Author grouping reduces one leakage risk but does not guarantee geographic independence. Acoustic detections and activity summaries describe recorded sound and are not direct estimates of animal population size or abundance.
