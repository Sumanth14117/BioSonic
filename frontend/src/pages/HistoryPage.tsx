import { useEffect, useState } from 'react'
import { AudioLines, ArrowRight, Clock3, RefreshCw } from 'lucide-react'
import { Link } from 'react-router'
import { PageHeading } from '../components/PageHeading'
import { getPredictionHistory } from '../lib/api'
import type { StoredPrediction } from '../types/api'

function prettyDate(epochSeconds: number) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(epochSeconds * 1000))
}

export function HistoryPage() {
  const [items, setItems] = useState<StoredPrediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true); setError('')
    try { setItems((await getPredictionHistory()).items) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Prediction history could not be loaded.') } finally { setLoading(false) }
  }

  useEffect(() => {
    let active = true
    getPredictionHistory().then((result) => { if (active) setItems(result.items) })
      .catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Prediction history could not be loaded.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  return <div className="page-stack">
    <PageHeading eyebrow="PRIVATE WORKSPACE · HISTORY" title="Prediction history" description="Saved inference summaries associated with your account. Uploaded audio is not stored." action={<button className="button-quiet" type="button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} /> Refresh</button>} />
    {error && <div className="inline-alert alert-error" role="alert">{error}</div>}
    {loading ? <div className="page-loading" role="status"><span className="spinner" /> Loading account history…</div> : items.length ? <section className="panel history-list" aria-label="Saved predictions">{items.map((item) => <Link className="history-row" to={`/results/${item.id}`} key={item.id}><span className="recent-icon"><AudioLines size={17} /></span><span className="history-main"><strong>{item.file_name}</strong><small><Clock3 size={12} /> {prettyDate(item.created_at)}</small></span><span className="history-class"><small>TOP CLASS</small><strong>{item.response.prediction}</strong></span><span className="recent-confidence">{Math.round(item.response.confidence * 100)}%</span><ArrowRight size={15} className="recent-arrow" /></Link>)}</section> : <section className="panel empty-state"><span className="empty-icon"><AudioLines size={22} /></span><strong>No account predictions yet</strong><p>Your new analyses will appear here. Any older browser-only history has not been imported or linked to this account and remains untouched on this device.</p><Link className="button-primary" to="/analysis">Analyse a recording <AudioLines size={16} /></Link></section>}
  </div>
}
