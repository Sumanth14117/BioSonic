# BioSonic — AI-Powered Wildlife Monitoring & Biodiversity Intelligence System

BioSonic is a research application exploring how machine learning can help inspect wildlife audio. Its current API supports local recording-level bird-audio classification. A separate soundscape study explored multi-species interval detection, but that work remains experimental and is not exposed as a reliable detector.

## Project status

| Milestone | Focus | Status |
|---|---|---|
| M1 | Research, environment setup, and dataset understanding | Complete |
| M2 | Audio preprocessing and exploratory analysis | Complete |
| M3 | Traditional machine learning | Complete for the interim experiment |
| M4 | Custom CNN development and evaluation | Complete for the interim experiment |
| M5 | Transfer learning and model comparison | Complete for the interim experiment |
| M6 | Model optimization | Complete as a training-only study |
| M7 | Multi-species soundscape detection | Exploratory feasibility study; reliable detection was not established |
| M8 | FastAPI backend and model API integration | Complete as a locally verified research API |
| M9 | Testing, deployment, documentation, and portfolio release | In progress; no deployment claimed |
| M10.1–M10.4 | Local authentication, dashboard, signal analysis, and batch analysis | Implemented; browser review remains pending |
| M10.5 | Final integration, QA, deployment readiness, and presentation | In progress; local research application only |

M3–M6 and the M4 API use a limited interim class set because the approved BirdCLEF acquisition is incomplete. Their evaluations reused a previously evaluated holdout, so cross-model comparisons are not independent external estimates. M7 is a separate multi-label soundscape task; its exploratory results do not establish useful detection performance. Acoustic activity is not a direct measure of wildlife population size or population change.

## Application

The local React interface supports account registration and login, audio upload, M4 predictions, signal visualizations, prediction history, batch comparison, and account settings. The FastAPI service validates OGG/Vorbis input and runs the existing M4 checkpoint. M7 is not available as an API prediction endpoint. The classifier always returns a label from its trained set; it does not reject unknown species or represent simultaneous species in a recording.

M10 authentication stores account/session data in a local SQLite database under ignored `data/`. Passwords use salted scrypt hashes; sessions are server-side and CSRF/Origin checks protect state-changing routes. This is local research functionality, not a hosted identity service or production security claim. Legacy browser-local prediction history is not migrated to accounts. Uploaded audio is not retained by the API.

The browser’s signal views are exploratory and separate from model preprocessing. M3/M4 use the experiment’s Mel pipeline; M5 uses YAMNet’s own waveform frontend after in-memory resampling. These representations are not interchangeable. The application does not make ecological presence, abundance, or population estimates.

## Environments and local setup

The project was developed with separate Windows environments because the main PyTorch/CUDA stack and native-Windows TensorFlow/YAMNet stack use different Python versions. The dependency files are snapshots of the existing environments; a fresh-clone installation from them has not been verified.

Main ML environment: Python 3.14.7, PyTorch 2.15.0+cu132, torchaudio 2.11.1+cu132, torchvision 0.29.2+cu132, and CUDA 13.2. CUDA was verified on an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM. Install commands from the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch==2.15.0+cu132" "torchaudio==2.11.1+cu132" "torchvision==0.29.2+cu132" --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements-main.txt
```

For the local API, install its declared optional dependencies in the same environment:

```powershell
python -m pip install -r requirements-api.txt
```

The isolated YAMNet environment uses Python 3.12.10 and TensorFlow/tf-keras 2.21.0. In this native-Windows setup, YAMNet runs on CPU rather than the RTX GPU:

```powershell
py -3.12 -m venv .venv-yamnet
& .\.venv-yamnet\Scripts\python.exe -m pip install -r requirements-yamnet.txt
```

The API additionally requires the packages in `requirements-api.txt` and a locally available M4 checkpoint. Neither the Python environments nor model checkpoints are included. To start the API from the repository root:

```powershell
\.venv\Scripts\python.exe scripts\run_m8_api.py
```

In another PowerShell window, start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

The app is served locally at `http://127.0.0.1:5173`; the API listens at `http://127.0.0.1:8000`. The frontend uses the Vite development proxy. These are local development instructions, not deployment steps. Browser-driven visual review and fresh-clone installation remain outstanding.

## Tests

Frontend checks, from `frontend/`:

```powershell
npm test -- --run
npm run lint
npm run build
```

The frontend lint command uses Oxlint. Backend checks use the existing `.venv`; YAMNet tests require `.venv-yamnet`. See `docs/api/usage.md` for API verification notes and test commands. These checks do not constitute browser visual sign-off or deployment readiness.

## Project structure

| Path | Purpose |
|---|---|
| `src/biosonic_ml/` | Recording-level classical ML and soundscape research workflows. |
| `src/biosonic_cnn/` | Custom CNN audio features and training pipeline. |
| `src/biosonic_yamnet/` | Isolated YAMNet embedding and transfer-learning workflow. |
| `src/biosonic_api/` | FastAPI application, upload validation, auth, and M4 inference integration. |
| `frontend/` | React/TypeScript local interface. |
| `scripts/` | Acquisition and experiment entry points. |
| `tests/` | Backend and model-workflow regression tests. |
| `docs/` | Preprocessing, API, authentication, architecture, and demo guidance. |
| `data/` | Local datasets and derived data; ignored by Git. |
| `models/` | Local checkpoints and generated classifiers; weights are not included. |

## Documentation

- [API usage](docs/api/usage.md)
- [Authentication and account storage](docs/authentication.md)
- [Frontend architecture](docs/frontend_architecture.md)
- [Frontend setup](frontend/README.md)
- [Demo guidance](docs/demo_guidance.md)
- [YAMNet provenance](models/m5_yamnet/PROVENANCE.md)

Aggregate research reports and generated dataset-derived figures are outside the scope of this conservative source-code publication set. Some research reports and figures from earlier milestones are already tracked in prior commits; this update does not remove or rewrite those files or their repository history.

## Dataset and third-party provenance

BirdCLEF+ 2026 is the source dataset. Its competition data is subject to [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the [competition terms](https://www.kaggle.com/competitions/birdclef-2026/data). Dataset-derived material requires BirdCLEF+ 2026 and contributor attribution, license identification, and change indication where applicable. Attribution alone does not resolve whether a particular derived report or result may be redistributed under the NonCommercial and ShareAlike terms. BioSonic does not claim ownership of the recordings or dataset-derived material.

The root [LICENSE](LICENSE) applies Apache License 2.0 only to original BioSonic source code. It does not automatically license BirdCLEF-derived material or third-party components. M5 uses Google’s YAMNet; provenance and the unresolved independent checkpoint-license verification are documented in [YAMNet provenance](models/m5_yamnet/PROVENANCE.md). Vendored YAMNet source and the AudioSet class map retain their separate upstream terms.

Dataset audio and metadata, model weights, generated feature caches, and local databases are not included in this source-code publication set. Do not infer permission to redistribute them from the BioSonic source license.
