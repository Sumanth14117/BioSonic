import type { PredictionResponse } from '../types/api'

export type BatchStatus = 'queued' | 'validating' | 'uploading' | 'processing' | 'completed' | 'failed' | 'cancelled'

export interface BatchExportRow {
  fileName: string
  status: BatchStatus
  response?: PredictionResponse
  timestamp?: string
}

function csvCell(value: string | number | undefined): string {
  let text = value === undefined ? '' : String(value)
  if (typeof value === 'string') {
    const first = text.trimStart().charAt(0)
    if (first === '=' || first === '+' || first === '-' || first === '@' || first.charCodeAt(0) < 32) text = `'${text}`
  }
  return `"${text.replaceAll('"', '""')}"`
}

export function batchCsv(rows: BatchExportRow[]): string {
  const header = ['filename', 'predicted_class', 'confidence', 'model_id', 'model_version', 'duration_seconds', 'analysis_status', 'timestamp']
  return [header, ...rows.map((row) => [
    row.fileName,
    row.response?.prediction,
    row.response?.confidence,
    row.response?.model,
    row.response?.model_version,
    row.response?.audio.duration_seconds,
    row.status,
    row.timestamp,
  ])].map((line) => line.map(csvCell).join(',')).join('\r\n') + '\r\n'
}

export function batchJson(rows: BatchExportRow[]): string {
  return JSON.stringify({ exported_at: new Date().toISOString(), items: rows.map((row) => ({
    filename: row.fileName,
    status: row.status,
    timestamp: row.timestamp ?? null,
    prediction_response: row.response ?? null,
  })) }, null, 2)
}
