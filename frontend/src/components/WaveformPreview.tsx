import { useEffect, useState } from 'react'
import { AudioLines } from 'lucide-react'

function createWavePath(samples: Float32Array, bars: number, width: number, height: number): string {
  const center = height / 2
  const step = Math.max(1, Math.floor(samples.length / bars))
  const paths: string[] = []
  for (let index = 0; index < bars; index += 1) {
    const start = index * step
    const end = Math.min(samples.length, start + step)
    let peak = 0
    for (let cursor = start; cursor < end; cursor += 1) peak = Math.max(peak, Math.abs(samples[cursor] ?? 0))
    const amplitude = Math.max(2, peak * (height * 0.43))
    const x = (index / bars) * width
    paths.push(`M${x.toFixed(1)} ${(center - amplitude).toFixed(1)} V${(center + amplitude).toFixed(1)}`)
  }
  return paths.join(' ')
}

export function WaveformPreview({ file }: { file: File }) {
  const [decoded, setDecoded] = useState<{ file: File; path: string; unavailable: boolean } | null>(null)
  const wavePath = decoded?.file === file ? decoded.path : ''
  const unavailable = decoded?.file === file && decoded.unavailable

  useEffect(() => {
    let active = true
    let context: AudioContext | undefined
    let path = ''
    let failed = false

    async function decode() {
      try {
        const AudioContextType = window.AudioContext
        if (!AudioContextType) throw new Error('Web Audio unavailable')
        context = new AudioContextType()
        const buffer = await file.arrayBuffer()
        const decoded = await context.decodeAudioData(buffer)
        path = createWavePath(decoded.getChannelData(0), 240, 960, 120)
      } catch {
        failed = true
      } finally {
        if (active) setDecoded({ file, path, unavailable: failed })
        await context?.close().catch(() => undefined)
      }
    }

    void decode()
    return () => { active = false; void context?.close().catch(() => undefined) }
  }, [file])

  return (
    <div className="waveform-panel">
      <div className="waveform-heading"><span><AudioLines size={15} /> Waveform preview</span><small>Browser decoded · not a model feature</small></div>
      {wavePath ? (
        <svg className="waveform-svg" viewBox="0 0 960 120" role="img" aria-label="Waveform visualization of the selected audio">
          <defs><linearGradient id="waveFill" x1="0" x2="1"><stop offset="0%" stopColor="#6ecb8e" /><stop offset="100%" stopColor="#b4e8b9" /></linearGradient></defs>
          <path d={wavePath} fill="none" stroke="url(#waveFill)" strokeWidth="2.4" strokeLinecap="round" opacity=".92" />
          <line x1="0" y1="60" x2="960" y2="60" stroke="#456052" strokeWidth="1" strokeDasharray="3 8" />
        </svg>
      ) : (
        <div className="waveform-empty">{unavailable ? 'Waveform preview is unavailable in this browser. Audio playback and server-side validation remain available.' : <><span className="wave-skeleton" /><span className="wave-skeleton short" /><span className="wave-skeleton" /></>}</div>
      )}
    </div>
  )
}
