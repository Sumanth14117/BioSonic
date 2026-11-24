import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { Link } from 'react-router'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Ban, Check, Download, FileAudio2, FileUp, Play, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import { useAuth } from '../components/AuthContext'
import { PageHeading } from '../components/PageHeading'
import { ApiError, requestPrediction, validateUploadFile } from '../lib/api'
import { batchCsv, batchJson, type BatchStatus } from '../lib/batchReport'
import type { PredictionResponse } from '../types/api'

interface QueueItem { id: string; file: File; status: BatchStatus; error?: string; response?: PredictionResponse; timestamp?: string; progress?: number }
const CONCURRENCY = 2

function keyFor(file: File) { return `${file.name.toLowerCase()}|${file.size}|${file.lastModified}` }
function saveDownload(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const link = document.createElement('a'); link.href = url; link.download = name; link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

export function BatchAnalysisPage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const itemsRef = useRef<QueueItem[]>([])
  const busyRef = useRef(false)
  const controllers = useRef(new Map<string, AbortController>())
  const [items, setItems] = useState<QueueItem[]>([])
  const [errors, setErrors] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const { csrfToken } = useAuth()

  function update(id: string, patch: Partial<QueueItem>) {
    const next = itemsRef.current.map((item) => item.id === id ? { ...item, ...patch } : item)
    itemsRef.current = next; setItems(next)
  }

  function addFiles(files: FileList | File[]) {
    const additions: QueueItem[] = [], problems: string[] = []
    const seen = new Set(itemsRef.current.map((item) => keyFor(item.file)))
    for (const file of Array.from(files)) {
      const error = validateUploadFile(file)
      if (error) { problems.push(`${file.name}: ${error}`); continue }
      const key = keyFor(file)
      if (seen.has(key)) { problems.push(`${file.name}: duplicate selection ignored.`); continue }
      seen.add(key); additions.push({ id: crypto.randomUUID(), file, status: 'queued' })
    }
    const next = [...itemsRef.current, ...additions]
    itemsRef.current = next; setItems(next); setErrors(problems)
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) { if (event.currentTarget.files) addFiles(event.currentTarget.files); event.currentTarget.value = '' }
  function onDrop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files) }

  async function processOne(id: string) {
    const item = itemsRef.current.find((candidate) => candidate.id === id)
    if (!item || !['queued', 'failed'].includes(item.status)) return
    update(id, { status: 'validating', error: undefined, progress: undefined })
    const error = validateUploadFile(item.file)
    if (error) { update(id, { status: 'failed', error }); return }
    const controller = new AbortController(); controllers.current.set(id, controller)
    update(id, { status: 'uploading' })
    try {
      const response = await requestPrediction(item.file, csrfToken, (percentage) => update(id, { progress: percentage, status: percentage >= 100 ? 'processing' : 'uploading' }), controller.signal)
      update(id, { status: 'completed', response, timestamp: new Date().toISOString(), progress: undefined })
    } catch (reason) {
      if (controller.signal.aborted || (reason instanceof ApiError && reason.code === 'aborted')) update(id, { status: 'cancelled', error: 'Request cancelled. The server may finish processing an upload already received.' })
      else update(id, { status: 'failed', error: reason instanceof ApiError ? reason.message : 'This recording could not be analyzed.' })
    } finally { controllers.current.delete(id) }
  }

  async function run(ids?: string[]) {
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    const selected = ids ?? itemsRef.current.filter((item) => item.status === 'queued').map((item) => item.id)
    let cursor = 0
    const worker = async () => { while (cursor < selected.length) { const id = selected[cursor++]; await processOne(id) } }
    try { await Promise.all(Array.from({ length: Math.min(CONCURRENCY, selected.length) }, worker)) }
    finally { busyRef.current = false; setBusy(false) }
  }

  function cancel() {
    for (const controller of controllers.current.values()) controller.abort()
    const next = itemsRef.current.map((item) => item.status === 'queued' ? { ...item, status: 'cancelled' as const } : item)
    itemsRef.current = next; setItems(next)
  }

  function remove(id: string) { const next = itemsRef.current.filter((item) => item.id !== id); itemsRef.current = next; setItems(next) }
  function clearCompleted() { const next = itemsRef.current.filter((item) => !['completed', 'cancelled'].includes(item.status)); itemsRef.current = next; setItems(next) }
  const completed = items.filter((item) => item.status === 'completed' && item.response)
  const failed = items.filter((item) => item.status === 'failed' || item.status === 'cancelled')
  const chartData = completed.map((item) => ({ name: item.file.name, confidence: item.response!.confidence, duration: item.response!.audio.duration_seconds, prediction: item.response!.prediction }))

  return <div className="page-stack batch-page">
    <PageHeading eyebrow="RECORDING WORKFLOW · M10.4" title="Batch audio analysis" description="Queue multiple recordings and submit them through the same authenticated M4 prediction API, with two requests at most in flight." />
    <section className="panel batch-upload-panel">
      <div className={`dropzone batch-dropzone${dragging ? ' dropzone-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <input ref={inputRef} className="visually-hidden" type="file" accept=".ogg,audio/ogg" multiple onChange={onInput} aria-label="Choose multiple OGG audio files" />
        <span className="upload-symbol"><Upload size={23} /></span><strong>Drop OGG recordings here</strong><span>or add several local files at once</span><button className="button-secondary" type="button" onClick={() => inputRef.current?.click()}><FileUp size={16} /> Browse recordings</button><small>Each file uses the existing 25 MiB, 1–180 second, mono 32 kHz OGG/Vorbis API contract.</small>
      </div>
      {errors.length > 0 && <ul className="batch-errors" role="alert">{errors.map((error, index) => <li key={`${index}-${error}`}>{error}</li>)}</ul>}
    </section>
    <section className="panel batch-queue-panel" aria-labelledby="batch-queue-title">
      <div className="panel-heading"><div><span className="eyebrow">CURRENT BATCH</span><h3 id="batch-queue-title">Recording queue <small>{items.length} files</small></h3></div><div className="batch-actions"><button className="button-secondary" type="button" onClick={() => void run()} disabled={busy || !items.some((item) => item.status === 'queued')}><Play size={15} /> Start queue</button>{busy && <button className="button-secondary" type="button" onClick={cancel}><Ban size={15} /> Cancel requests</button>}<button className="button-icon" type="button" aria-label="Clear completed queue items" onClick={clearCompleted}><Trash2 size={16} /></button></div></div>
      {!items.length ? <div className="batch-empty"><FileAudio2 size={21} /><strong>No recordings queued</strong><p>Add OGG files to create a batch comparison.</p></div> : <div className="batch-list">{items.map((item) => <article className="batch-item" key={item.id}>
        <div className="batch-file-icon"><FileAudio2 size={17} /></div><div className="batch-file-main"><strong title={item.file.name}>{item.file.name}</strong><small>{(item.file.size / 1024 / 1024).toFixed(2)} MiB · {item.status}{item.status === 'processing' ? ' · waiting for API response' : ''}{item.progress !== undefined && item.status === 'uploading' ? ` · upload ${item.progress}%` : ''}</small>{item.error && <small className="batch-item-error">{item.error}</small>}{item.response && <small className="batch-item-result">{item.response.prediction} · {Math.round(item.response.confidence * 100)}% score · {item.response.model} v{item.response.model_version}</small>}</div>
        <span className={`batch-status status-${item.status}`}>{item.status === 'completed' ? <Check size={13} /> : null}{item.status}</span>
        {item.response && <details className="batch-score-details"><summary aria-label={`Show eight class scores for ${item.file.name}`}>Scores</summary><ol>{Object.entries(item.response.probabilities).sort((a, b) => b[1] - a[1]).map(([label, score]) => <li key={label}><code>{label}</code><span>{(score * 100).toFixed(1)}%</span></li>)}</ol></details>}
        {item.response && <Link className="button-icon" aria-label={`View result for ${item.file.name}`} to={`/results/${item.response.prediction_id}`}>↗</Link>}
        {item.status === 'failed' && <button className="button-icon" aria-label={`Retry ${item.file.name}`} onClick={() => void run([item.id])} disabled={busy}><RefreshCw size={15} /></button>}
        {item.status === 'queued' && <button className="button-icon" aria-label={`Remove ${item.file.name}`} onClick={() => remove(item.id)}><X size={15} /></button>}
      </article>)}</div>}
      <p className="batch-cancel-note">Cancellation stops the browser request. If the server has already received the audio, inference may finish and be recorded in your account history.</p>
    </section>
    <section className="panel batch-results-panel"><div className="panel-heading"><div><span className="eyebrow">BATCH COMPARISON</span><h3>Actual completed predictions</h3></div><div className="batch-actions"><button className="button-secondary" type="button" disabled={!items.length} onClick={() => saveDownload('biosonic-batch-report.csv', batchCsv(items.map((item) => ({ fileName: item.file.name, status: item.status, response: item.response, timestamp: item.timestamp }))), 'text/csv;charset=utf-8')}><Download size={15} /> Export CSV</button><button className="button-secondary" type="button" disabled={!items.length} onClick={() => saveDownload('biosonic-batch-report.json', batchJson(items.map((item) => ({ fileName: item.file.name, status: item.status, response: item.response, timestamp: item.timestamp }))), 'application/json')}><Download size={15} /> Export JSON</button></div></div>
      <p className="batch-counts">{completed.length} completed · {failed.length} failed or cancelled · {items.length - completed.length - failed.length} pending/in progress</p>
      {!completed.length ? <div className="batch-empty"><strong>No completed predictions yet</strong><p>Comparison charts use successful responses in this batch only.</p></div> : <>
        <div className="batch-chart-grid"><div><h4>Confidence score by recording</h4><div className="batch-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid stroke="#2a382d" vertical={false} /><XAxis dataKey="name" tick={{ fill: '#87998b', fontSize: 9 }} hide /><YAxis domain={[0, 1]} tick={{ fill: '#87998b', fontSize: 9 }} /><Tooltip /><Bar dataKey="confidence" fill="#83d49a" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div><p className="signal-method">Scores are model confidence outputs, not calibrated probabilities.</p></div><div><h4>Predicted class counts</h4><div className="batch-class-list">{Object.entries(completed.reduce<Record<string, number>>((counts, item) => { const label = item.response!.prediction; counts[label] = (counts[label] ?? 0) + 1; return counts }, {})).sort((a, b) => b[1] - a[1]).map(([label, count]) => <div key={label}><code>{label}</code><strong>{count}</strong></div>)}</div></div></div>
        <div className="batch-result-table"><div className="batch-table-head"><span>Recording</span><span>Prediction</span><span>Confidence</span><span>Duration</span><span>Model</span></div>{completed.map((item) => <div className="batch-table-row" key={item.id}><span title={item.file.name}>{item.file.name}</span><span>{item.response!.prediction}</span><span>{(item.response!.confidence * 100).toFixed(1)}%</span><span>{item.response!.audio.duration_seconds.toFixed(2)} s</span><span>{item.response!.model}</span></div>)}</div>
      </>}
      {failed.length > 0 && <div className="batch-failure-summary"><strong>Failures are isolated per recording</strong><span>{failed.map((item) => `${item.file.name}: ${item.error ?? item.status}`).join(' · ')}</span></div>}
    </section>
    <div className="notice-row"><div className="notice-icon"><Ban size={17} /></div><p><strong>Interpretation.</strong> Batch class counts summarize the model's returned labels in this upload only. They are not wildlife population, abundance, or biodiversity distribution estimates.</p></div>
  </div>
}
