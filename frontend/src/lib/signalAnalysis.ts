export interface SignalAnalysis {
  sampleRate: number
  channels: number
  sampleCount: number
  duration: number
  mean: number
  rms: number
  peak: number
  silencePercent: number
  silenceThreshold: number
  clipping099Percent: number
  clipping1Percent: number
  waveformMin: number[]
  waveformMax: number[]
  waveformTimes: number[]
  nFft: number
  hopLength: number
  frequencyResolution: number
  timeResolution: number
  frequencies: number[]
  spectrum: number[]
  dominantFrequency: number | null
  frameTimes: number[]
  stftDb: number[]
  stftFrequencyBins: number
  melDb: number[]
  melBands: number
}

const FFT_SIZE = 1024
const HOP = 256
const MEL_BANDS = 128
const MAX_FRAMES = 1000

function fftPower(frame: Float64Array): Float64Array {
  const n = frame.length
  const re = new Float64Array(frame)
  const im = new Float64Array(n)
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]] }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len
    const wr0 = Math.cos(angle), wi0 = Math.sin(angle)
    for (let base = 0; base < n; base += len) {
      let wr = 1, wi = 0
      for (let j = 0; j < len / 2; j++) {
        const even = base + j, odd = even + len / 2
        const tr = wr * re[odd] - wi * im[odd], ti = wr * im[odd] + wi * re[odd]
        re[odd] = re[even] - tr; im[odd] = im[even] - ti
        re[even] += tr; im[even] += ti
        const nextWr = wr * wr0 - wi * wi0
        wi = wr * wi0 + wi * wr0; wr = nextWr
      }
    }
  }
  const output = new Float64Array(n / 2 + 1)
  for (let i = 0; i < output.length; i++) output[i] = (re[i] * re[i] + im[i] * im[i]) / (n * n)
  return output
}

function hzToMel(hz: number): number {
  const fSp = 200 / 3
  return hz < 1000 ? hz / fSp : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27)
}

function melToHz(mel: number): number {
  const fSp = 200 / 3
  return mel < 15 ? mel * fSp : 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15))
}

function melFilters(sampleRate: number): { bin: number; weight: number }[][] {
  const maxHz = Math.min(sampleRate / 2, 16000)
  const lowMel = hzToMel(0), highMel = hzToMel(maxHz)
  const points = Array.from({ length: MEL_BANDS + 2 }, (_, i) => melToHz(lowMel + (highMel - lowMel) * i / (MEL_BANDS + 1)))
  const filters: { bin: number; weight: number }[][] = []
  for (let band = 0; band < MEL_BANDS; band++) {
    const left = points[band], center = points[band + 1], right = points[band + 2]
    const weights: { bin: number; weight: number }[] = []
    for (let bin = 0; bin <= FFT_SIZE / 2; bin++) {
      const hz = bin * sampleRate / FFT_SIZE
      const triangle = hz < center ? (hz - left) / (center - left || 1) : (right - hz) / (right - center || 1)
      const weight = Math.max(0, triangle) * (2 / (right - left || 1))
      if (weight > 0) weights.push({ bin, weight })
    }
    filters.push(weights)
  }
  return filters
}

export function downsampleWaveform(samples: Float32Array, sampleRate: number, maxPoints = 1200) {
  if (!samples.length || !Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error('Audio contains no valid samples.')
  const buckets = Math.min(maxPoints, samples.length)
  const min: number[] = [], max: number[] = [], times: number[] = []
  for (let bucket = 0; bucket < buckets; bucket++) {
    const start = Math.floor(bucket * samples.length / buckets)
    const end = Math.max(start + 1, Math.floor((bucket + 1) * samples.length / buckets))
    let lo = Infinity, hi = -Infinity
    for (let i = start; i < end; i++) { lo = Math.min(lo, samples[i]); hi = Math.max(hi, samples[i]) }
    min.push(lo); max.push(hi); times.push((start + end) / 2 / sampleRate)
  }
  return { min, max, times }
}

export function analyzeSignal(samples: Float32Array, sampleRate: number, channels = 1): SignalAnalysis {
  if (!samples.length || !Number.isInteger(sampleRate) || sampleRate <= 0) throw new Error('Audio contains no valid samples.')
  let sum = 0, squareSum = 0, peak = 0, at099 = 0, at1 = 0
  for (const value of samples) {
    if (!Number.isFinite(value)) throw new Error('Audio contains a non-finite sample.')
    sum += value; squareSum += value * value; peak = Math.max(peak, Math.abs(value))
    if (Math.abs(value) >= 0.99) at099++
    if (Math.abs(value) >= 1) at1++
  }
  const mean = sum / samples.length
  const rms = Math.sqrt(squareSum / samples.length)
  const silenceFrame = Math.max(1, Math.round(sampleRate * 0.1)), silenceHop = Math.max(1, Math.round(sampleRate * 0.05))
  let quiet = 0, frames = 0
  for (let start = 0; start < samples.length; start += silenceHop) {
    const end = Math.min(samples.length, start + silenceFrame)
    let localSquares = 0
    for (let i = start; i < end; i++) localSquares += samples[i] * samples[i]
    if (Math.sqrt(localSquares / (end - start)) < 0.005) quiet++
    frames++
  }
  const wave = downsampleWaveform(samples, sampleRate)
  const frameCount = samples.length < FFT_SIZE ? 0 : Math.floor((samples.length - FFT_SIZE) / HOP) + 1
  const stride = Math.max(1, Math.ceil(frameCount / MAX_FRAMES))
  const filters = melFilters(sampleRate)
  const selectedTimes: number[] = [], stftFrames: number[][] = [], melFrames: number[][] = []
  const spectrumSum = new Float64Array(FFT_SIZE / 2 + 1)
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += stride) {
    const start = frameIndex * HOP
    const windowed = new Float64Array(FFT_SIZE)
    for (let i = 0; i < FFT_SIZE; i++) windowed[i] = samples[start + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1)))
    const power = fftPower(windowed)
    for (let bin = 0; bin < power.length; bin++) spectrumSum[bin] += Math.sqrt(power[bin])
    stftFrames.push(Array.from(power))
    melFrames.push(filters.map((filter) => {
      let energy = 0
      for (const entry of filter) energy += entry.weight * power[entry.bin]
      return energy
    }))
    selectedTimes.push((start + FFT_SIZE / 2) / sampleRate)
  }
  const frequencies = Array.from({ length: FFT_SIZE / 2 + 1 }, (_, i) => i * sampleRate / FFT_SIZE)
  const spectrum = Array.from(spectrumSum, (value) => value / Math.max(stftFrames.length, 1))
  let dominantIndex = 1
  for (let i = 2; i < spectrum.length; i++) if (spectrum[i] > spectrum[dominantIndex]) dominantIndex = i
  let maxPower = 1e-12, maxMel = 1e-12
  for (const frame of stftFrames) for (const value of frame) maxPower = Math.max(maxPower, value)
  for (const frame of melFrames) for (const value of frame) maxMel = Math.max(maxMel, value)
  const db = (v: number, reference: number) => Math.max(-80, 10 * Math.log10(Math.max(v, 1e-12) / reference))
  return {
    sampleRate, channels, sampleCount: samples.length, duration: samples.length / sampleRate, mean, rms, peak,
    silencePercent: frames ? quiet / frames * 100 : 0, silenceThreshold: 0.005,
    clipping099Percent: at099 / samples.length * 100, clipping1Percent: at1 / samples.length * 100,
    waveformMin: wave.min, waveformMax: wave.max, waveformTimes: wave.times,
    nFft: FFT_SIZE, hopLength: HOP, frequencyResolution: sampleRate / FFT_SIZE, timeResolution: HOP / sampleRate,
    frequencies, spectrum, dominantFrequency: frameCount && peak > 1e-8 ? frequencies[dominantIndex] : null,
    frameTimes: selectedTimes,
    stftDb: stftFrames.flatMap((frame) => frame.map((v) => db(v, maxPower))), stftFrequencyBins: FFT_SIZE / 2 + 1,
    melDb: melFrames.flatMap((frame) => frame.map((v) => db(v, maxMel))), melBands: MEL_BANDS,
  }
}
