import { useEffect, useState, type CSSProperties } from 'react'
import { Link, useParams } from 'react-router'
import { ArrowLeft, AudioLines, CircleHelp, Clock3, FileAudio2, Info, ShieldAlert } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PageHeading } from '../components/PageHeading'
import { getPrediction } from '../lib/api'
import type { StoredPrediction } from '../types/api'

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return minutes ? `${minutes}m ${remainder.toFixed(1)}s` : `${remainder.toFixed(2)}s`
}

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function ResultsPage() {
  const { id } = useParams()
  const [record, setRecord] = useState<StoredPrediction | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    getPrediction(id ?? '').then((item) => { if (active) setRecord(item) }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'This result could not be loaded.')
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [id])

  if (loading) return <div className="page-loading" role="status"><span className="spinner" /> Loading saved prediction…</div>

  if (!record) {
    return <div className="page-stack"><PageHeading eyebrow="ANALYSIS · RESULTS" title="Result unavailable" description={error || 'Run an audio analysis to see a saved result here.'} /><div className="panel empty-state"><span className="empty-icon"><AudioLines size={22} /></span><strong>This prediction is not available to this account</strong><p>Saved results are private to the account that created them.</p><Link className="button-primary" to="/analysis">Analyse a recording <AudioLines size={16} /></Link></div></div>
  }

  const { response } = record
  const chartData = Object.entries(response.probabilities).map(([label, value]) => ({ label, value })).sort((left, right) => right.value - left.value)

  return (
    <div className="page-stack">
      <PageHeading eyebrow="ANALYSIS · RESULT DETAIL" title="Prediction results" description="A recording-level summary from the M4 model. Review the class scores in context." action={<Link className="button-quiet" to="/analysis"><ArrowLeft size={15} /> New analysis</Link>} />

      <section className="result-hero panel">
        <div className="result-main"><div className="result-label"><span className="result-check"><AudioLines size={16} /></span><span className="eyebrow">TOP PREDICTED CLASS</span></div><h2>{response.prediction}</h2><p>The highest model score among its trained classes.</p><div className="result-tags"><span><span className="result-tag-dot" /> {response.model}</span><span>Version {response.model_version}</span></div></div>
        <div className="confidence-ring" style={{ '--confidence-target': `${response.confidence * 100}%` } as CSSProperties} role="img" aria-label={`Model confidence ${Math.round(response.confidence * 100)} percent`}><div className="ring-inner"><strong>{Math.round(response.confidence * 100)}<small>%</small></strong><span>CONFIDENCE</span></div></div>
      </section>

      <div className="results-grid">
        <section className="panel probability-panel"><div className="panel-heading"><div><span className="eyebrow">CLASS SCORES · RANKED</span><h3>Class probabilities</h3></div><span className="panel-meta">Softmax probability</span></div><div className="probability-chart" role="img" aria-label="Ranked probability bars for the model's trained classes"><ResponsiveContainer width="100%" height={330}><BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="#23352a" />
          <XAxis type="number" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fill: '#809187', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="label" width={88} tick={{ fill: '#d9e5dc', fontSize: 12, fontFamily: 'ui-monospace, monospace' }} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: '#1b2a20' }} contentStyle={{ background: '#17221b', border: '1px solid #32483a', borderRadius: 10, color: '#edf5ee' }} formatter={(value) => [`${(Number(value) * 100).toFixed(2)}%`, 'Score']} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} fill="#78cb91" maxBarSize={19} />
        </BarChart></ResponsiveContainer></div><div className="chart-footnote"><Info size={14} /> Scores are model outputs, not calibrated probabilities of biological presence.</div></section>

        <section className="panel recording-panel"><div className="panel-heading"><div><span className="eyebrow">RECORDING DETAILS</span><h3>Input summary</h3></div><FileAudio2 size={18} className="heading-icon" /></div><div className="recording-name"><span className="recording-file-icon"><AudioLines size={18} /></span><span><strong title={record.file_name}>{record.file_name}</strong><small>Analysed {new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(record.created_at * 1000))}</small></span></div><div className="detail-stack"><div className="detail-row"><span>Duration</span><strong>{formatDuration(response.audio.duration_seconds)}</strong></div><div className="detail-row"><span>Sample rate</span><strong>{(response.audio.sample_rate / 1000).toFixed(0)} kHz</strong></div><div className="detail-row"><span>Channels</span><strong>{response.audio.channels === 1 ? 'Mono' : response.audio.channels}</strong></div><div className="detail-row"><span>Decoded format</span><strong>{response.audio.format}</strong></div><div className="detail-row"><span>File size</span><strong>{formatSize(record.file_size)}</strong></div><div className="detail-row"><span>Segment windows</span><strong>{Math.ceil(response.audio.duration_seconds / 5)}</strong></div></div><div className="model-version"><Clock3 size={14} /> Model version <code>{response.model_version}</code></div><div className="result-playback-unavailable"><FileAudio2 size={16} /><span><strong>Original audio is not retained</strong>Reopen the source recording locally if you need playback.</span></div></section>
      </div>

      <div className="notice-row result-notice"><div className="notice-icon"><ShieldAlert size={17} /></div><p><strong>Confidence is not certainty.</strong> This closed-set model always returns one of its trained labels. A high score does not confirm a species is present, and a low score does not mean it is absent. Review the recording and consider expert validation.</p><CircleHelp size={15} className="notice-end-icon" />
      </div>
      <div className="result-bottom-link"><Link className="text-link" to="/models">Read model limitations <ArrowLeft size={14} /></Link></div>
    </div>
  )
}
