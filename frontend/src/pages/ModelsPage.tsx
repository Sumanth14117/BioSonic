import { useEffect, useState } from 'react'
import { Activity, ArrowUpRight, AudioLines, BadgeCheck, Cpu, FlaskConical, Info, Layers3, Sparkles, Waves } from 'lucide-react'
import { getModels } from '../lib/api'
import type { ApiModel } from '../types/api'
import { PageHeading } from '../components/PageHeading'

type ExperimentModel = {
  milestone: string
  title: string
  method: string
  status: 'integrated' | 'research' | 'experimental'
  context: string
  icon: typeof Cpu
}

const experiments: ExperimentModel[] = [
  { milestone: 'M3', title: 'Traditional ML', method: 'RBF SVM on recording-level audio features', status: 'research', context: 'Interim study using a previously evaluated holdout also used in later experiments; results are not independent comparisons.', icon: Layers3 },
  { milestone: 'M4', title: 'Custom CNN', method: 'Four-stage CNN on normalized 128-band Mel inputs', status: 'integrated', context: 'The checkpoint currently exposed by the API. It predicts from a fixed trained label set and has no unknown-species rejection.', icon: Cpu },
  { milestone: 'M5', title: 'YAMNet transfer learning', method: 'Frozen YAMNet embeddings + Logistic Regression', status: 'research', context: 'Runs through an isolated TensorFlow CPU workflow; it is not available from the current prediction endpoint.', icon: AudioLines },
  { milestone: 'M6', title: 'Model optimization', method: 'Training-only grouped CV on frozen embeddings', status: 'research', context: 'The shared holdout was not accessed during this optimization study.', icon: Activity },
  { milestone: 'M7', title: 'Soundscape feasibility', method: 'Site-grouped multi-label interval study', status: 'experimental', context: 'Experimental/research-only work. Reliable wildlife detection was not established, and this study is disabled in the API.', icon: Waves },
]

export function ModelsPage() {
  const [apiModels, setApiModels] = useState<ApiModel[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    getModels().then((result) => setApiModels(result.models)).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Model inventory is unavailable.'))
  }, [])

  const integrated = apiModels.find((model) => model.id === 'biosonic-m4-cnn')
  const m7Api = apiModels.find((model) => model.id === 'biosonic-m7-soundscape')

  return (
    <div className="page-stack">
      <PageHeading eyebrow="EXPERIMENT REGISTRY · M3—M7" title="Models & methods" description="A transparent map of the project’s experimental methods, limitations, and what the current API can actually run." />
      {error && <div className="inline-alert alert-info"><Info size={16} />{error} Experiment method descriptions are still shown below.</div>}
      <div className="model-registry">
        {experiments.map((item) => {
          const Icon = item.icon
          const isIntegrated = item.milestone === 'M4'
          const actualStatus = isIntegrated ? integrated?.status : item.milestone === 'M7' ? m7Api?.status : undefined
          return <article key={item.milestone} className={`panel experiment-card${isIntegrated ? ' experiment-integrated' : ''}`}>
            <div className="experiment-top"><span className={`milestone-chip milestone-${item.status}`}>{item.milestone}</span><span className={`experiment-status ${isIntegrated ? 'status-active' : item.status === 'experimental' ? 'status-experimental' : 'status-not-api'}`}><span className="status-dot" />{isIntegrated ? `API ${actualStatus ?? 'status unknown'}` : item.milestone === 'M7' ? `API ${actualStatus ?? 'disabled'}` : 'Research only'}</span></div>
            <div className="experiment-title"><span className="experiment-icon"><Icon size={19} /></span><div><h3>{item.title}</h3><p>{item.method}</p></div></div>
            <p className="experiment-context">{item.context}</p>
            {isIntegrated && integrated && <div className="api-model-classes"><span>API classes</span><div>{integrated.classes.map((label) => <code key={label}>{label}</code>)}</div><small><BadgeCheck size={13} /> {integrated.device ?? 'Device unavailable'} · version {integrated.version ?? 'n/a'}</small></div>}
            {item.milestone === 'M7' && <div className="api-model-classes disabled-note"><span>API availability</span><small><Info size={13} /> This capability is intentionally disabled; no soundscape endpoint is exposed.</small></div>}
          </article>
        })}
      </div>
      <div className="notice-row"><div className="notice-icon"><Sparkles size={17} /></div><p><strong>Results are not directly comparable.</strong> M3–M5 reused the same previously evaluated holdout, while M6 used training-only grouped cross-validation. M7 is a separate task and evaluation design.</p><ArrowUpRight size={15} className="notice-end-icon" /></div>
      <div className="small-footnote"><FlaskConical size={14} /> Reported metrics describe past experiments. This page does not evaluate or rank a model at runtime.</div>
    </div>
  )
}
