# BioSonic demo and screenshot guidance

No screenshots or demo video are currently claimed by this guide. Capture real application screens only after browser review, using an account and audio recording approved for demonstration. Do not include passwords, email addresses, session cookies, CSRF values, local profile paths, recording author identities, or private prediction history in public captures.

## Suggested capture sequence

1. **Sign in / registration:** use a synthetic demo account; crop away browser password-manager UI.
2. **Overview:** show live API/model readiness and empty history or clearly labeled account-derived charts.
3. **Single-file analysis:** show the upload contract and actual local recording selection; obscure identifying filenames if needed.
4. **Waveform and playback:** capture the decoded waveform with time/seek controls and its zoom setting.
5. **STFT spectrogram:** include the displayed FFT/hop/window and relative dB assumptions.
6. **Mel spectrogram:** include filter configuration and clarify that it is exploratory, not M4 model input.
7. **Prediction detail:** show actual returned model/class scores and explain that the classifier is closed-set.
8. **Authenticated analytics/history:** use only account-owned predictions and avoid implying abundance or population estimates.
9. **Batch queue and comparison:** show actual completed and failed rows; class counts describe only that uploaded batch.
10. **Model information and About:** show model provenance, experiment limitations, and research-only status.

## Viewport review

Check the navigation drawer, upload controls, canvas legibility, batch table horizontal overflow, chart tooltips, and accessible focus at desktop, tablet, and mobile widths. Verify keyboard operation of file selection, playback controls, waveform seek slider, zoom/speed selectors, queue actions, and result links. Respect `prefers-reduced-motion`. Do not describe CSS inspection or unit tests as browser visual verification.

## Demo video

No demo video is included. If one is later recorded, use genuine application output, disclose the limited local research scope, and avoid presenting model predictions as confirmed species presence or direct wildlife population estimates.
