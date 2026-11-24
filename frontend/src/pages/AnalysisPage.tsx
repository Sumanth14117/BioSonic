import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { useNavigate } from 'react-router'
import { AudioLines, Check, FileAudio2, FileUp, Info, LoaderCircle, ShieldCheck, Upload, X } from 'lucide-react'
import { PageHeading } from '../components/PageHeading'
import { AudioIntelligence } from '../components/AudioIntelligence'
import { ApiError, requestPrediction, validateUploadFile } from '../lib/api'
import { useAuth } from '../components/AuthContext'
import { useToast } from '../lib/toastContext'

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function AnalysisPage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState('')
  const [requestError, setRequestError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const navigate = useNavigate()
  const { csrfToken } = useAuth()
  const { showToast } = useToast()
  const previewUrl = useMemo(() => file ? URL.createObjectURL(file) : '', [file])

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  function chooseFile(candidate?: File) {
    if (!candidate) return
    const validationMessage = validateUploadFile(candidate)
    setRequestError('')
    setProgress(null)
    if (validationMessage) {
      setFile(null)
      setFileError(validationMessage)
      return
    }
    setFileError('')
    setFile(candidate)
  }

  function onFileInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.currentTarget.files?.[0])
    event.currentTarget.value = ''
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  async function analyze() {
    if (!file || busy) return
    setBusy(true)
    setRequestError('')
    setProgress(null)
    try {
      const response = await requestPrediction(file, csrfToken, setProgress)
      showToast('Analysis complete. Results are ready to review.', 'success')
      navigate(`/results/${response.prediction_id}`)
    } catch (error) {
      const message = error instanceof ApiError ? error.message : 'The analysis could not be completed. Try again.'
      setRequestError(message)
      showToast(message, 'error')
    } finally {
      setBusy(false)
    }
  }

  const progressLabel = progress === null ? 'Preparing recording upload' : progress >= 100 ? 'Upload complete · waiting for prediction' : `Uploading recording · ${progress}%`

  return (
    <div className="page-stack">
      <PageHeading eyebrow="RECORDING WORKFLOW · M4" title="Analyse a recording" description="Upload one mono 32 kHz OGG/Vorbis recording for a real M4 model prediction." />

      <div className="analysis-layout">
        <section className="panel upload-panel" aria-labelledby="upload-title">
          <div className="panel-heading upload-heading"><div><span className="eyebrow">AUDIO INPUT</span><h3 id="upload-title">Choose a field recording</h3></div><span className="format-badge"><FileAudio2 size={14} /> OGG / VORBIS</span></div>

          {!file ? (
            <div className={`dropzone${dragging ? ' dropzone-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <input ref={inputRef} className="visually-hidden" type="file" accept=".ogg,audio/ogg" onChange={onFileInput} aria-label="Choose OGG audio file" />
              <div className="upload-symbol"><Upload size={23} /></div>
              <strong>Drop an audio file here</strong>
              <span>or choose it from your device</span>
              <button className="button-secondary" type="button" onClick={() => inputRef.current?.click()}><FileUp size={16} /> Browse files</button>
              <small>OGG/Vorbis · up to 25 MiB · 1–180 seconds · mono, 32 kHz</small>
            </div>
          ) : (
            <div className="selected-audio">
              <div className="file-row"><span className="file-icon"><AudioLines size={19} /></span><div className="file-details"><strong title={file.name}>{file.name}</strong><span>{formatSize(file.size)} <i>·</i> Waiting for analysis</span></div><button className="button-icon remove-file" aria-label="Remove selected audio" onClick={() => { setFile(null); setRequestError(''); setFileError('') }} disabled={busy}><X size={17} /></button></div>
              <AudioIntelligence file={file} previewUrl={previewUrl} />
            </div>
          )}

          {fileError && <div className="inline-alert alert-error" role="alert"><X size={16} />{fileError}</div>}
          {requestError && <div className="inline-alert alert-error" role="alert"><Info size={16} /><span>{requestError}</span></div>}

          {busy && <div className="upload-progress" role="status" aria-live="polite"><div className="progress-label"><span><LoaderCircle size={15} className="spin" /> {progressLabel}</span>{progress !== null && progress < 100 && <span>{progress}%</span>}</div><div className="progress-track"><span className={progress === null ? 'progress-indeterminate' : ''} style={progress === null ? undefined : { width: `${progress}%` }} /></div></div>}

          <button className="button-primary analyze-button" type="button" onClick={() => void analyze()} disabled={!file || busy}>
            {busy ? <><LoaderCircle size={17} className="spin" /> Analysing recording…</> : <><AudioLines size={17} /> Analyse audio <span className="button-arrow">→</span></>}
          </button>
        </section>

        <aside className="analysis-aside">
          <section className="panel contract-panel"><span className="eyebrow">INPUT CONTRACT</span><h3>Before you upload</h3><ul className="contract-list"><li><Check size={15} /> OGG/Vorbis container</li><li><Check size={15} /> Mono · 32,000 Hz</li><li><Check size={15} /> 1 to 180 seconds</li><li><Check size={15} /> Maximum 25 MiB</li></ul><div className="contract-note"><ShieldCheck size={16} /><p>Audio is processed by your local API and not retained. Your prediction summary is saved to your account history.</p></div></section>
          <section className="panel pipeline-mini"><span className="eyebrow">WHAT HAPPENS NEXT</span><div className="pipeline-step"><span>01</span><div><strong>Validate</strong><small>Format, duration, channels, sample rate</small></div></div><div className="pipeline-step"><span>02</span><div><strong>Prepare</strong><small>M4 five-second windows and Mel inputs</small></div></div><div className="pipeline-step"><span>03</span><div><strong>Classify</strong><small>Eight-class probabilities at recording level</small></div></div></section>
        </aside>
      </div>
      <div className="notice-row"><div className="notice-icon"><Info size={17} /></div><p><strong>Scope reminder.</strong> This interim classifier always chooses among its trained bird classes; it cannot reject unknown species or identify simultaneous species.</p></div>
    </div>
  )
}
