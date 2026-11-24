import { describe, expect, it } from 'vitest'
import type { StoredPrediction } from '../types/api'
import { buildHistoryAnalytics } from './analytics'

function prediction(id: string, createdAt: number, label: string, duration: number): StoredPrediction {
  return {
    id, file_name: `${id}.ogg`, file_size: 100, created_at: createdAt,
    response: {
      prediction_id: id, model: 'biosonic-m4-cnn', model_version: 'test', prediction: label,
      confidence: 0.7, probabilities: { [label]: 0.7, other: 0.3 },
      audio: { duration_seconds: duration, sample_rate: 32_000, channels: 1, sample_count: duration * 32_000, format: 'OGG/VORBIS' },
    },
  }
}

describe('history analytics', () => {
  it('aggregates only supplied account history and returns newest recent items', () => {
    const analytics = buildHistoryAnalytics([
      prediction('a', 1_735_689_600, 'burowl', 4),
      prediction('b', 1_735_776_000, 'burowl', 12),
      prediction('c', 1_735_776_001, 'chobla1', 40),
    ])
    expect(analytics.recent.map((item) => item.id)).toEqual(['c', 'b', 'a'])
    expect(analytics.predictionsByDay).toEqual([{ date: '2025-01-01', count: 1 }, { date: '2025-01-02', count: 2 }])
    expect(analytics.classCounts).toEqual([{ label: 'burowl', count: 2 }, { label: 'chobla1', count: 1 }])
    expect(analytics.durationBands.map(({ count }) => count)).toEqual([1, 1, 1, 0])
    expect(analytics.averageDurationSeconds).toBeCloseTo(56 / 3)
  })

  it('returns empty chart data and no averages when history is empty', () => {
    const analytics = buildHistoryAnalytics([])
    expect(analytics.recent).toEqual([])
    expect(analytics.predictionsByDay).toEqual([])
    expect(analytics.classCounts).toEqual([])
    expect(analytics.durationBands.every(({ count }) => count === 0)).toBe(true)
    expect(analytics.averageDurationSeconds).toBeNull()
  })
})
