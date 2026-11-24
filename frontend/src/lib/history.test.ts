import { describe, expect, it } from 'vitest'
import { clearHistory, loadHistory, saveHistory } from './history'
import type { PredictionRecord } from '../types/api'

function record(id: string): PredictionRecord {
  return {
    id, analyzedAt: '2026-11-23T00:00:00.000Z', fileName: `${id}.ogg`, fileSize: 128,
    response: { prediction_id: id, model: 'biosonic-m4-cnn', model_version: 'a1b2', prediction: 'burowl', confidence: 0.8, probabilities: { burowl: 0.8 }, audio: { duration_seconds: 5, sample_rate: 32000, channels: 1, sample_count: 160000, format: 'OGG/VORBIS' } },
  }
}

describe('local analysis history', () => {
  it('stores summaries in most-recent-first order without audio bytes', () => {
    saveHistory(record('first'))
    const updated = saveHistory(record('second'))
    expect(updated.map((item) => item.id)).toEqual(['second', 'first'])
    expect(loadHistory()[0]?.fileName).toBe('second.ogg')
    expect(JSON.stringify(updated)).not.toContain('ogg bytes')
  })

  it('ignores invalid stored JSON and can clear local history', () => {
    localStorage.setItem('biosonic.analysis-history.v1', '{invalid')
    expect(loadHistory()).toEqual([])
    saveHistory(record('sample'))
    clearHistory()
    expect(loadHistory()).toEqual([])
  })
})
