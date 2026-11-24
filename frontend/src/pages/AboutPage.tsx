import { ArrowRight, AudioLines, Bird, Compass, Ear, Leaf, ShieldAlert, Waves } from 'lucide-react'
import { Link } from 'react-router'
import { PageHeading } from '../components/PageHeading'

const stages = [
  { code: '01', title: 'Load & validate', detail: 'Local audio upload is checked for OGG/Vorbis, mono, 32 kHz, duration and decodability.', icon: Ear },
  { code: '02', title: 'Build model input', detail: 'M4 forms five-second windows and computes its training-compatible Mel representation.', icon: AudioLines },
  { code: '03', title: 'Infer & aggregate', detail: 'The existing CNN scores each window; scores are averaged into one recording-level result.', icon: Waves },
  { code: '04', title: 'Review evidence', detail: 'Model outputs are shown for human review. The interface makes no ecological abundance estimate.', icon: Compass },
]

export function AboutPage() {
  return (
    <div className="page-stack">
      <PageHeading eyebrow="PROJECT · PURPOSE & LIMITS" title="About BioSonic" description="A research project exploring how machine learning can help make wildlife audio easier to inspect and analyze." />

      <section className="about-hero panel"><div className="about-emblem"><span><Bird size={36} strokeWidth={1.35} /></span><i className="emblem-ring ring-a" /><i className="emblem-ring ring-b" /></div><div><span className="eyebrow">AI-POWERED WILDLIFE MONITORING</span><h2>Useful signals.<br /><span>Honest uncertainty.</span></h2><p>BioSonic’s current interface connects a local FastAPI service to an interim bird recording classifier. It is a prototype for research and learning, not a field-ready detection or conservation decision system.</p></div></section>

      <section className="panel pipeline-panel"><div className="panel-heading"><div><span className="eyebrow">CURRENT INFERENCE FLOW</span><h3>From recording to review</h3></div><span className="panel-meta">M4 API path</span></div><div className="about-pipeline">{stages.map(({ code, title, detail, icon: Icon }) => <article className="about-step" key={code}><span className="step-code">{code}</span><span className="step-icon"><Icon size={18} /></span><h4>{title}</h4><p>{detail}</p></article>)}</div></section>

      <div className="about-grid">
        <section className="panel about-card"><div className="about-card-icon"><Bird size={18} /></div><span className="eyebrow">EXPERIMENTAL SCOPE</span><h3>Bird-audio research</h3><p>BioSonic is a research application for bird-audio analysis and machine-learning experimentation. Its experiments use limited datasets and do not establish a field-ready wildlife detector.</p></section>
        <section className="panel about-card"><div className="about-card-icon"><Leaf size={18} /></div><span className="eyebrow">PIPELINE MILESTONES</span><h3>M1 through M8</h3><p>Dataset understanding, audio exploration, traditional ML, a custom CNN, YAMNet transfer learning, optimization, a limited soundscape feasibility study, and a locally verified API.</p><Link className="text-link" to="/models">Review the model registry <ArrowRight size={14} /></Link></section>
      </div>

      <section className="limitations-panel"><div className="limitations-icon"><ShieldAlert size={18} /></div><div><span className="eyebrow">SCIENTIFIC LIMITATIONS</span><h3>Audio activity is not animal abundance.</h3><p>A classifier score is not a confirmed species observation. The current endpoint predicts from its trained labels and cannot reject unknown species or represent overlapping species. Recordings, sites, devices and annotator behavior can affect results. Acoustic detections and activity summaries are not direct measures of population size or population change.</p></div></section>

      <div className="about-cta"><span>Have a compatible recording?</span><Link className="button-primary" to="/analysis">Open audio analysis <AudioLines size={16} /></Link></div>
    </div>
  )
}
