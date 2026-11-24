import { useEffect, useRef, useState } from 'react'
import { Activity, AudioLines, Gauge, Waves } from 'lucide-react'
import type { SignalAnalysis } from '../lib/signalAnalysis'

function fmtTime(seconds: number) {
  if (!Number.isFinite(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  return `${mins}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

function drawHeatmap(canvas: HTMLCanvasElement, values: number[], rows: number, columns: number) {
  const width = canvas.clientWidth || 640, height = 210, ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio)
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.scale(ratio, ratio)
  ctx.fillStyle = '#111a14'; ctx.fillRect(0, 0, width, height)
  if (!columns || !rows) return
  const image = ctx.createImageData(columns, rows)
  for (let frame = 0; frame < columns; frame++) for (let displayRow = 0; displayRow < rows; displayRow++) {
    const sourceIndex = frame * rows + (rows - 1 - displayRow)
    const offset = (displayRow * columns + frame) * 4
    const strength = Math.max(0, Math.min(1, (values[sourceIndex] + 80) / 80))
    image.data[offset] = Math.round(18 + strength * 104)
    image.data[offset + 1] = Math.round(40 + strength * 185)
    image.data[offset + 2] = Math.round(34 + strength * 91)
    image.data[offset + 3] = 255
  }
  const scratch = document.createElement('canvas'); scratch.width = columns; scratch.height = rows
  scratch.getContext('2d')?.putImageData(image, 0, 0)
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(scratch, 0, 0, width, height)
}

function WaveformCanvas({ analysis, currentTime, duration, onSeek, zoom }: { analysis: SignalAnalysis; currentTime: number; duration: number; onSeek: (time: number) => void; zoom: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const start = Math.max(0, Math.min(duration * (1 - 1 / zoom), currentTime - duration / (2 * zoom)))
  const end = Math.min(duration, start + duration / zoom)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const width = canvas.clientWidth || 640, height = 120, ratio = window.devicePixelRatio || 1
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(ratio, ratio); ctx.fillStyle = '#111a14'; ctx.fillRect(0, 0, width, height)
    const mid = height / 2
    ctx.strokeStyle = '#26382c'; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(width, mid); ctx.stroke()
    ctx.strokeStyle = '#83d49a'; ctx.lineWidth = 1
    ctx.beginPath()
    const count = analysis.waveformMin.length
    for (let i = 0; i < count; i++) {
      const time = analysis.waveformTimes[i]
      if (time < start || time > end) continue
      const x = (time - start) / Math.max(end - start, 1e-9) * width
      ctx.moveTo(x, mid - analysis.waveformMax[i] * mid * .85)
      ctx.lineTo(x, mid - analysis.waveformMin[i] * mid * .85)
    }
    ctx.stroke()
    const marker = (currentTime - start) / Math.max(end - start, 1e-9) * width
    if (marker >= 0 && marker <= width) { ctx.strokeStyle = '#f0ca7a'; ctx.beginPath(); ctx.moveTo(marker, 0); ctx.lineTo(marker, height); ctx.stroke() }
  }, [analysis, currentTime, end, start])
  return <div className="signal-canvas-wrap"><canvas ref={canvasRef} className="signal-canvas waveform-canvas" aria-label="Decoded waveform amplitude over time" onClick={(event) => { const rect = event.currentTarget.getBoundingClientRect(); onSeek(start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * (end - start)) }} /><div className="axis-caption"><span>{fmtTime(start)}</span><span>Amplitude · decoded PCM</span><span>{fmtTime(end)}</span></div></div>
}

function Spectrum({ analysis }: { analysis: SignalAnalysis }) {
  const max = Math.max(...analysis.spectrum, 1e-9)
  const points = analysis.spectrum.map((value, index) => `${index / Math.max(analysis.spectrum.length - 1, 1) * 100},${100 - value / max * 95}`).join(' ')
  return <svg className="spectrum-svg" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Mean magnitude spectrum from zero to Nyquist frequency"><title>Mean magnitude by frequency, 0 to {analysis.sampleRate / 2} Hz. Dominant bin: {analysis.dominantFrequency ?? 'unavailable'} Hz.</title><polyline points={points} fill="none" stroke="#8ddb9d" strokeWidth=".7" vectorEffect="non-scaling-stroke" /></svg>
}

export function AudioIntelligence({ file, previewUrl }: { file: File; previewUrl: string }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const stftRef = useRef<HTMLCanvasElement>(null)
  const melRef = useRef<HTMLCanvasElement>(null)
  const [analysis, setAnalysis] = useState<SignalAnalysis | null>(null)
  const supported = typeof AudioContext !== 'undefined' && typeof Worker !== 'undefined'
  const [error, setError] = useState(supported ? '' : 'This browser does not support local audio analysis.')
  const [currentTime, setCurrentTime] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [processing, setProcessing] = useState(supported)

  useEffect(() => {
    let active = true
    let worker: Worker | null = null
    let context: AudioContext | null = null
    let setupErrorTimer = 0
    if (!supported) return () => { active = false }
    try {
      worker = new Worker(new URL('../lib/signalAnalysis.worker.ts', import.meta.url), { type: 'module' })
      context = new AudioContext()
    } catch (reason) {
      setupErrorTimer = window.setTimeout(() => {
        if (active) {
          setError(reason instanceof Error ? reason.message : 'Local audio analysis could not be started.')
          setProcessing(false)
        }
      }, 0)
      return () => { active = false; window.clearTimeout(setupErrorTimer); worker?.terminate(); void context?.close().catch(() => undefined) }
    }
    worker.onmessage = (event: MessageEvent<{ result?: SignalAnalysis; error?: string }>) => {
      if (!active) return
      if (event.data.error) setError(event.data.error)
      else if (event.data.result) setAnalysis(event.data.result)
      setProcessing(false)
    }
    void file.arrayBuffer().then((buffer) => context!.decodeAudioData(buffer)).then((decoded) => {
      if (!active) return
      const mono = new Float32Array(decoded.length)
      for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
        const values = decoded.getChannelData(channel)
        for (let i = 0; i < values.length; i++) mono[i] += values[i] / decoded.numberOfChannels
      }
      worker.postMessage({ samples: mono, sampleRate: decoded.sampleRate, channels: decoded.numberOfChannels }, [mono.buffer])
    }).catch(() => { if (active) { setError('This browser could not decode the selected audio for signal analysis.'); setProcessing(false) } }).finally(() => { void context?.close().catch(() => undefined) })
    return () => { active = false; worker?.terminate(); void context?.close().catch(() => undefined) }
  }, [file, supported])

  useEffect(() => { if (audioRef.current) audioRef.current.playbackRate = speed }, [speed])
  useEffect(() => {
    if (stftRef.current && analysis) drawHeatmap(stftRef.current, analysis.stftDb, analysis.stftFrequencyBins, analysis.frameTimes.length)
    if (melRef.current && analysis) drawHeatmap(melRef.current, analysis.melDb, analysis.melBands, analysis.frameTimes.length)
  }, [analysis])

  const duration = analysis?.duration ?? 0
  const format = file.name.toLowerCase().endsWith('.ogg') ? 'OGG extension · decoded' : 'Unknown'
  return <section className="signal-workspace panel" aria-labelledby="signal-title">
    <div className="panel-heading"><div><span className="eyebrow">LOCAL SIGNAL ANALYSIS</span><h3 id="signal-title">Audio intelligence workspace</h3></div><span className="format-badge"><Activity size={14} /> EXPLORATORY</span></div>
    <audio ref={audioRef} className="audio-player" controls preload="metadata" src={previewUrl} aria-label="Preview selected audio" onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onEnded={() => setCurrentTime(duration)} />
    <div className="signal-playback-controls"><label>Playback speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1">1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><label>Waveform zoom<select value={zoom} onChange={(event) => setZoom(Number(event.target.value))}><option value="1">Full recording</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label><span>{fmtTime(currentTime)} / {fmtTime(duration)}</span></div>
    {processing && <p className="signal-status" role="status"><span className="spinner" /> Decoding and measuring the selected recording locally…</p>}
    {error && <p className="inline-alert alert-error" role="alert">{error}</p>}
    {analysis && <>
      <div className="signal-block"><div className="signal-section-title"><Waves size={15} /><h4>Waveform</h4><small>Click the trace to seek; the displayed envelope is downsampled.</small></div><WaveformCanvas analysis={analysis} currentTime={currentTime} duration={duration} zoom={zoom} onSeek={(time) => { if (audioRef.current) audioRef.current.currentTime = time; setCurrentTime(time) }} /></div>
      <label className="waveform-seek">Seek audio position<input aria-label="Seek audio position" type="range" min="0" max={Math.max(duration, 0.01)} step="0.01" value={Math.min(currentTime, duration)} onChange={(event) => { const time = Number(event.target.value); if (audioRef.current) audioRef.current.currentTime = time; setCurrentTime(time) }} /></label>
      <div className="signal-metadata-grid">{[['File', file.name], ['Format', format], ['File size', `${(file.size / 1024 / 1024).toFixed(2)} MiB`], ['Decoded duration', `${duration.toFixed(2)} s`], ['Analysis sample rate', `${analysis.sampleRate.toLocaleString()} Hz (browser decode)`], ['Decoded channels', `${analysis.channels} · averaged for signal views`], ['Bit depth', 'Unavailable from browser-decoded PCM']].map(([label, value]) => <div key={label}><small>{label}</small><strong title={value}>{value}</strong></div>)}</div>
      <div className="signal-measurements">{[['Mean amplitude', analysis.mean.toFixed(5)], ['RMS amplitude', analysis.rms.toFixed(5)], ['Peak amplitude', analysis.peak.toFixed(5)], ['Low-energy frames', `${analysis.silencePercent.toFixed(1)}%`], ['Samples ≥ |0.99|', `${analysis.clipping099Percent.toFixed(4)}%`], ['Samples ≥ |1.0|', `${analysis.clipping1Percent.toFixed(4)}%`]].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div>
      <p className="signal-caveat">Low-energy estimate: 100 ms RMS frames, 50 ms hop, threshold RMS &lt; {analysis.silenceThreshold}. Clipping thresholds are diagnostics, not proof of clipped source audio. These are acoustic signal measurements, not species presence or abundance measurements.</p>
      <div className="signal-block"><div className="signal-section-title"><Activity size={15} /><h4>Linear-frequency STFT</h4><small>Power in relative dB; visualization only, not M4's model input.</small></div><div className="spectrogram-frame"><span className="frequency-axis">{(analysis.sampleRate / 2 / 1000).toFixed(1)} kHz<br />0 Hz</span><canvas ref={stftRef} className="signal-canvas stft-canvas" aria-label="STFT power spectrogram from zero to Nyquist frequency" /><span className="db-scale">0 dB<br />−80 dB</span></div><div className="axis-caption"><span>0 s</span><span>Time · seconds</span><span>{duration.toFixed(1)} s</span></div><p className="signal-method">NFFT {analysis.nFft} · hop {analysis.hopLength} ({(analysis.timeResolution * 1000).toFixed(2)} ms) · Hann window · frequency resolution {analysis.frequencyResolution.toFixed(2)} Hz · {analysis.frameTimes.length} displayed frames × {analysis.stftFrequencyBins} bins · power 10 log₁₀(P / max P), floor −80 dB.</p></div>
      <div className="signal-block"><div className="signal-section-title"><AudioLines size={15} /><h4>Mel spectrogram</h4><small>Perceptual Mel filter-bank view of the same STFT power.</small></div><div className="spectrogram-frame"><span className="frequency-axis">Mel<br />0</span><canvas ref={melRef} className="signal-canvas mel-canvas" aria-label="128-band Slaney Mel power spectrogram" /><span className="db-scale">0 dB<br />−80 dB</span></div><div className="axis-caption"><span>0 s</span><span>Time · seconds</span><span>{duration.toFixed(1)} s</span></div><p className="signal-method">128 triangular filters · Slaney Mel spacing and area normalization · 0–min(16 kHz, Nyquist) · {analysis.frameTimes.length} displayed frames × {analysis.melBands} bands · power dB relative to this recording's maximum. Mel spacing allocates more detail at lower frequencies; the model's internal M4 frontend is separate.</p></div>
      <div className="signal-block"><div className="signal-section-title"><Gauge size={15} /><h4>Mean frequency spectrum</h4><small>Average magnitude across analyzed STFT frames.</small></div><div className="spectrum-plot"><Spectrum analysis={analysis} /></div><div className="axis-caption"><span>0 Hz</span><span>Frequency</span><span>{(analysis.sampleRate / 2 / 1000).toFixed(1)} kHz</span></div><p className="signal-method">Dominant bin estimate: {analysis.dominantFrequency === null ? 'not available (recording is shorter than one FFT frame)' : `${analysis.dominantFrequency.toFixed(1)} Hz`}. A frequency peak alone cannot identify a species.</p></div>
    </>}
  </section>
}
