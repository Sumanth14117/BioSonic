import { analyzeSignal } from './signalAnalysis'

self.onmessage = (event: MessageEvent<{ samples: Float32Array; sampleRate: number; channels: number }>) => {
  try {
    self.postMessage({ result: analyzeSignal(event.data.samples, event.data.sampleRate, event.data.channels) })
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : 'Audio analysis failed.' })
  }
}
