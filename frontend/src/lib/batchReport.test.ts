import { describe, expect, it } from 'vitest'
import { batchCsv, batchJson, type BatchExportRow } from './batchReport'

describe('batch report exports', () => {
  const rows: BatchExportRow[] = [{ fileName: 'bird,"one".ogg', status: 'completed', timestamp: '2026-11-23T10:00:00Z', response: {
    model: 'biosonic-m4-cnn', model_version: 'abc123', prediction_id: 'private-id', prediction: 'banana', confidence: 0.8,
    probabilities: { banana: 0.8 }, audio: { duration_seconds: 5, sample_rate: 32000, channels: 1, sample_count: 160000, format: 'OGG/VORBIS' },
  } }, { fileName: '=cmd.ogg', status: 'failed' }]

  it('escapes CSV fields and emits failed rows without private path or token data', () => {
    const csv = batchCsv(rows)
    expect(csv).toContain('"bird,""one"".ogg"')
    expect(csv).toContain('"failed"')
    expect(csv).toContain("'=cmd.ogg")
    expect(csv).not.toContain('private-id')
    expect(csv).not.toContain('C:\\')
  })

  it('exports only current batch entries and full successful response JSON', () => {
    const json = JSON.parse(batchJson(rows)) as { items: { filename: string; prediction_response: { prediction_id: string } | null }[] }
    expect(json.items).toHaveLength(2)
    expect(json.items[0].prediction_response?.prediction_id).toBe('private-id')
    expect(json.items[1].prediction_response).toBeNull()
  })
})
