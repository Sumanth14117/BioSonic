import type { PredictionRecord } from '../types/api'

const STORAGE_KEY = 'biosonic.analysis-history.v1'
const MAX_RECORDS = 8

export function loadHistory(): PredictionRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isPredictionRecord).slice(0, MAX_RECORDS)
  } catch {
    return []
  }
}

function isPredictionRecord(value: unknown): value is PredictionRecord {
  if (!value || typeof value !== 'object') return false
  const item = value as Partial<PredictionRecord>
  return Boolean(
    typeof item.id === 'string' &&
    typeof item.analyzedAt === 'string' &&
    typeof item.fileName === 'string' &&
    typeof item.fileSize === 'number' &&
    item.response &&
    typeof item.response.prediction === 'string' &&
    item.response.probabilities &&
    typeof item.response.probabilities === 'object',
  )
}

export function saveHistory(record: PredictionRecord): PredictionRecord[] {
  const next = [record, ...loadHistory().filter((item) => item.id !== record.id)].slice(0, MAX_RECORDS)
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // Private browsing or a full storage quota should not block the current result.
  }
  return next
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // History is optional; a storage failure should not disrupt the application.
  }
}
