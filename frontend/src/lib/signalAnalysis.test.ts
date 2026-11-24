import { describe, expect, it } from 'vitest'
import { analyzeSignal, downsampleWaveform } from './signalAnalysis'

describe('browser signal analysis', () => {
  it('computes basic measurements and the expected dominant FFT bin', () => {
    const rate = 32_000
    const samples = Float32Array.from({ length: 8192 }, (_, index) => 0.5 * Math.sin(2 * Math.PI * 1000 * index / rate))
    const result = analyzeSignal(samples, rate, 1)
    expect(result.sampleCount).toBe(samples.length)
    expect(result.duration).toBeCloseTo(0.256)
    expect(result.rms).toBeCloseTo(Math.SQRT1_2 * 0.5, 2)
    expect(result.peak).toBeCloseTo(0.5, 2)
    expect(result.dominantFrequency).toBe(1000)
    expect(result.stftFrequencyBins).toBe(513)
    expect(result.melBands).toBe(128)
    expect(result.frameTimes.length).toBe(result.stftDb.length / 513)
    expect(result.frameTimes.length).toBe(result.melDb.length / 128)
  })

  it('bounds waveform rendering for long recordings while retaining local peaks', () => {
    const samples = new Float32Array(100_000).fill(0)
    samples[50_123] = 0.8
    const envelope = downsampleWaveform(samples, 32_000, 1000)
    expect(envelope.max).toHaveLength(1000)
    expect(Math.max(...envelope.max)).toBeCloseTo(0.8, 6)
  })

  it('rejects empty and non-finite decoded input', () => {
    expect(() => analyzeSignal(new Float32Array(), 32_000)).toThrow(/no valid samples/i)
    expect(() => analyzeSignal(Float32Array.of(0, Number.NaN), 32_000)).toThrow(/non-finite/i)
  })

  it('caps long-recording spectrogram frames and leaves a silent dominant frequency unavailable', () => {
    const long = analyzeSignal(new Float32Array(32_000 * 15), 32_000)
    expect(long.frameTimes.length).toBeLessThanOrEqual(1000)
    expect(long.waveformMin).toHaveLength(1200)
    expect(long.dominantFrequency).toBeNull()
    expect(long.stftDb).toHaveLength(long.frameTimes.length * 513)
  })
})
