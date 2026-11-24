import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, ArrowRight, AudioLines, BarChart3, Clock3, Leaf, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react'
import { Link } from 'react-router'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useAuth } from '../components/AuthContext'
import { PageHeading } from '../components/PageHeading'
import { getModels, getPredictionHistory, getReadiness } from '../lib/api'
import { buildHistoryAnalytics } from '../lib/analytics'
import type { ApiModel, StoredPrediction } from '../types/api'

type ApiStatus = 'checking' | 'ready' | 'degraded' | 'offline'

function displayName(email?: string) {
  const accountName = email?.split('@')[0]?.replace(/[._-]+/g, ' ').trim()
  if (!accountName) return 'there'
  return accountName.replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function EmptyChart({ title }: { title: string }) {
  return <div className="analytics-empty"><span><BarChart3 size={18} /></span><strong>{title}</strong><p>Save a few account predictions to see this chart.</p></div>
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: { value?: number; name?: string; color?: string }[]; label?: string }) {
  if (!active || !payload?.length) return null
  return <div className="chart-tooltip"><small>{label}</small>{payload.map((item) => <div key={item.name ?? 'value'}><i style={{ background: item.color }} />{item.name ?? 'Recordings'} <strong>{item.value ?? 0}</strong></div>)}</div>
}

async function fetchDashboardData() {
  const [systemResult, historyResult] = await Promise.allSettled([
    Promise.all([getReadiness(), getModels()]),
    getPredictionHistory(),
  ])
  return { systemResult, historyResult }
}

type DashboardData = Awaited<ReturnType<typeof fetchDashboardData>>

export function DashboardPage() {
  const { user } = useAuth()
  const [models, setModels] = useState<ApiModel[]>([])
  const [history, setHistory] = useState<StoredPrediction[]>([])
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [loading, setLoading] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [historyError, setHistoryError] = useState('')

  const applyDashboardData = useCallback(({ systemResult, historyResult }: DashboardData) => {
    if (systemResult.status === 'fulfilled') {
      const [readiness, inventory] = systemResult.value
      setModels(inventory.models)
      setApiStatus(readiness.status === 'ready' ? 'ready' : 'degraded')
    } else {
      setModels([])
      setApiStatus('offline')
      setLoadError('The inference service could not be reached. Start the local API and refresh.')
    }
    setLoading(false)

    if (historyResult.status === 'fulfilled') setHistory(historyResult.value.items)
    else {
      setHistory([])
      setHistoryError('Account history is unavailable. Try refreshing after confirming your session.')
    }
    setHistoryLoading(false)
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    setHistoryLoading(true)
    setLoadError('')
    setHistoryError('')
    applyDashboardData(await fetchDashboardData())
  }, [applyDashboardData])

  useEffect(() => {
    let active = true
    void fetchDashboardData().then((data) => { if (active) applyDashboardData(data) })
    return () => { active = false }
  }, [applyDashboardData])

  const analytics = useMemo(() => buildHistoryAnalytics(history), [history])
  const activeModel = models.find((model) => model.id === 'biosonic-m4-cnn')
  const topClass = analytics.classCounts[0]
  const statusText = apiStatus === 'ready' ? 'Connected and ready' : apiStatus === 'degraded' ? 'Connected · model unavailable' : apiStatus === 'offline' ? 'API unavailable' : 'Checking service'
  const savedCount = history.length

  return <div className="page-stack dashboard-page">
    <PageHeading eyebrow="FIELD INTELLIGENCE · WORKSPACE" title={`Good to see you, ${displayName(user?.email)}.`} description="A clear view of your local BioSonic analysis workspace." action={<button className="button-quiet refresh-button" onClick={() => void refresh()} disabled={loading || historyLoading} aria-label="Refresh dashboard data"><RefreshCw size={16} className={loading || historyLoading ? 'spin' : ''} /> Refresh</button>} />

    <section className="dashboard-hero">
      <div className="dashboard-hero-copy"><span className="welcome-kicker"><span className="live-mark" /> PRIVATE RESEARCH WORKSPACE</span><h2>Every recording<br /><em>adds context.</em></h2><p>Review model outputs alongside your own saved analyses. These predictions describe classifier behavior, not ecological abundance.</p><Link className="button-primary" to="/analysis"><AudioLines size={17} /> Analyse a recording <ArrowRight size={16} /></Link></div>
      <div className="dashboard-orbit" aria-hidden="true"><div className="orbit-ring orbit-ring-one" /><div className="orbit-ring orbit-ring-two" /><span className="orbit-satellite satellite-one" /><span className="orbit-satellite satellite-two" /><div className="orbit-center"><Activity size={29} /></div><small>WILDLIFE AUDIO</small></div>
    </section>

    <section className="overview-grid" aria-label="Workspace overview">
      <article className="overview-card metric-card"><span className="metric-label"><AudioLines size={15} /> Saved analyses</span><strong className="metric-value">{historyLoading ? '—' : savedCount}</strong><small>{historyLoading ? 'Loading account history' : 'Latest records available to this account'}</small></article>
      <article className="overview-card metric-card"><span className="metric-label"><Leaf size={15} /> Most predicted class</span><strong className="metric-value metric-class">{historyLoading ? '—' : topClass?.label ?? '—'}</strong><small>{historyLoading ? 'Loading account history' : topClass ? `${topClass.count} of ${savedCount} saved predictions` : 'Available after your first prediction'}</small></article>
      <article className="overview-card metric-card"><span className="metric-label"><Clock3 size={15} /> Mean recording length</span><strong className="metric-value">{historyLoading || analytics.averageDurationSeconds === null ? '—' : `${analytics.averageDurationSeconds.toFixed(1)} s`}</strong><small>{historyLoading ? 'Loading account history' : 'Calculated from available saved history'}</small></article>
      <article className={`overview-card status-card status-card-${apiStatus}`}><span className="metric-label"><span className="status-dot" /> API status</span><strong className="status-copy">{statusText}</strong><small>{loadError || (activeModel ? `${activeModel.id} · ${activeModel.version ?? 'version unavailable'}` : 'M4 model inventory not available')}</small></article>
    </section>

    <section className="model-summary-card panel">
      <div className="model-summary-top"><div className="model-mark"><Sparkles size={18} /></div><div className="model-heading"><span className="eyebrow">ACTIVE MODEL · LIVE INVENTORY</span><h3>{activeModel?.id ?? 'M4 model'}</h3><p>{activeModel?.task ?? 'Recording-level bird classification'}</p></div><span className={`model-state ${activeModel?.status === 'ready' ? 'model-state-ready' : ''}`}><i />{loading ? 'Checking' : activeModel?.status ?? 'Unavailable'}</span></div>
      {activeModel ? <><div className="model-class-list" aria-label="Model classes">{activeModel.classes.map((label) => <span key={label}>{label}</span>)}</div><div className="model-summary-bottom"><span>Version <code>{activeModel.version ?? 'not reported'}</code></span><span>Device <strong>{activeModel.device ?? 'not reported'}</strong></span><Link to="/models" className="text-link">Model details <ArrowRight size={14} /></Link></div></> : <div className="model-unavailable">{loadError || 'Model information appears when the API is available.'}</div>}
    </section>

    <section className="analytics-section" aria-label="Prediction analytics">
      <div className="analytics-heading"><div><span className="eyebrow">ACCOUNT ANALYTICS</span><h2>Your recent signal</h2></div><span>Built from saved predictions · max 100 records</span></div>
      {historyError && <div className="inline-alert alert-error" role="alert">{historyError}</div>}
      <div className="analytics-grid">
        <article className="panel chart-card chart-card-wide"><div className="chart-card-heading"><div><span className="eyebrow">ACTIVITY</span><h3>Predictions by day</h3></div><span className="chart-mark"><Activity size={15} /></span></div>{historyLoading ? <div className="chart-loading" role="status"><span className="spinner" /> Loading history…</div> : analytics.predictionsByDay.length ? <div className="chart-stage"><ResponsiveContainer width="100%" height="100%"><AreaChart data={analytics.predictionsByDay} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}><defs><linearGradient id="historyArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#83d49a" stopOpacity={0.28} /><stop offset="100%" stopColor="#83d49a" stopOpacity={0.015} /></linearGradient></defs><CartesianGrid stroke="#28382d" strokeDasharray="3 6" vertical={false} /><XAxis dataKey="date" tickFormatter={(value: string) => value.slice(5)} tick={{ fill: '#829487', fontSize: 10 }} axisLine={false} tickLine={false} minTickGap={18} /><YAxis allowDecimals={false} tick={{ fill: '#829487', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip content={<ChartTooltip />} /><Area type="monotone" dataKey="count" name="Predictions" stroke="#8ddba0" strokeWidth={2.2} fill="url(#historyArea)" activeDot={{ r: 4, fill: '#c0efc6', stroke: '#183021' }} isAnimationActive /></AreaChart></ResponsiveContainer></div> : <EmptyChart title="No activity yet" />}</article>

        <article className="panel chart-card"><div className="chart-card-heading"><div><span className="eyebrow">CLASS MIX</span><h3>Most predicted classes</h3></div><span className="chart-mark"><Leaf size={15} /></span></div>{historyLoading ? <div className="chart-loading" role="status"><span className="spinner" /> Loading history…</div> : analytics.classCounts.length ? <div className="chart-stage chart-stage-tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={analytics.classCounts.slice(0, 8)} layout="vertical" margin={{ top: 4, right: 10, left: 0, bottom: 0 }}><CartesianGrid stroke="#28382d" strokeDasharray="3 6" horizontal={false} /><XAxis type="number" allowDecimals={false} tick={{ fill: '#829487', fontSize: 9 }} axisLine={false} tickLine={false} /><YAxis dataKey="label" type="category" width={77} tick={{ fill: '#b5c5b8', fontSize: 10, fontFamily: 'ui-monospace, monospace' }} axisLine={false} tickLine={false} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="count" name="Predictions" fill="#8ddba0" radius={[0, 5, 5, 0]} maxBarSize={18} isAnimationActive /></BarChart></ResponsiveContainer></div> : <EmptyChart title="No class counts yet" />}</article>

        <article className="panel chart-card"><div className="chart-card-heading"><div><span className="eyebrow">RECORDING LENGTH</span><h3>Duration bands</h3></div><span className="chart-mark"><Clock3 size={15} /></span></div>{historyLoading ? <div className="chart-loading" role="status"><span className="spinner" /> Loading history…</div> : savedCount ? <div className="chart-stage chart-stage-compact"><ResponsiveContainer width="100%" height="100%"><BarChart data={analytics.durationBands} margin={{ top: 12, right: 4, left: -20, bottom: 0 }}><CartesianGrid stroke="#28382d" strokeDasharray="3 6" vertical={false} /><XAxis dataKey="band" tick={{ fill: '#829487', fontSize: 9 }} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tick={{ fill: '#829487', fontSize: 9 }} axisLine={false} tickLine={false} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="count" name="Recordings" fill="#c2ad77" radius={[5, 5, 0, 0]} maxBarSize={34} isAnimationActive /></BarChart></ResponsiveContainer></div> : <EmptyChart title="No duration data yet" />}</article>

        <article className="panel recent-activity-card"><div className="chart-card-heading"><div><span className="eyebrow">RECENT RECORDINGS</span><h3>Latest predictions</h3></div><Link to="/history" className="text-link">View history <ArrowRight size={14} /></Link></div>{historyLoading ? <div className="chart-loading" role="status"><span className="spinner" /> Loading history…</div> : analytics.recent.length ? <div className="recent-activity-list">{analytics.recent.map((item) => <Link className="recent-activity-row" to={`/results/${item.id}`} key={item.id}><span className="recent-icon"><AudioLines size={16} /></span><span className="recent-activity-file"><strong>{item.file_name}</strong><small>{new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(item.created_at * 1000))}</small></span><span className="recent-activity-class"><small>TOP CLASS</small><strong>{item.response.prediction}</strong></span><span className="recent-confidence">{Math.round(item.response.confidence * 100)}%</span><ArrowRight size={14} className="recent-arrow" /></Link>)}</div> : <div className="dashboard-history-empty"><span className="empty-icon"><AudioLines size={19} /></span><div><strong>No saved predictions yet</strong><p>Your account analytics will appear after an audio analysis.</p></div><Link className="text-link" to="/analysis">Start analysis <ArrowRight size={14} /></Link></div>}</article>
      </div>
    </section>

    <div className="notice-row"><div className="notice-icon"><ShieldAlert size={17} /></div><p><strong>Interpret with care.</strong> This closed-set model assigns each accepted recording to one of its trained classes. Prediction counts describe your saved model outputs, not wildlife presence trends or population estimates.</p></div>
  </div>
}
