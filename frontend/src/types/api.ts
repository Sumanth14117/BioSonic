export interface ProcessHealth {
  status: string
}

export interface ApiReadiness {
  status: 'ready' | 'degraded' | string
  models: Record<string, string>
}

export interface ApiModel {
  id: string
  task: string
  status: 'ready' | 'unavailable' | 'disabled' | string
  version: string | null
  classes: string[]
  limitations: string[]
  device: string | null
}

export interface ModelsResponse {
  models: ApiModel[]
}

export interface AudioMetadata {
  duration_seconds: number
  sample_rate: number
  channels: number
  sample_count: number
  format: string
}

export interface PredictionResponse {
  model: string
  model_version: string
  prediction_id: string
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  audio: AudioMetadata
}

export interface ApiErrorDetail {
  code: string
  message: string
}

export interface ApiErrorPayload {
  detail?: ApiErrorDetail | string
}

export interface LocalPredictionRecord {
  id: string
  analyzedAt: string
  fileName: string
  fileSize: number
  response: PredictionResponse
}

/** Legacy browser-only record type; M10.1 does not import or associate it with accounts. */
export type PredictionRecord = LocalPredictionRecord

export interface AuthUser {
  id: string
  email: string
}

export interface AuthStateResponse {
  csrf_token: string
  user: AuthUser | null
}

export interface StoredPrediction {
  id: string
  file_name: string
  file_size: number
  created_at: number
  response: PredictionResponse
}

export interface PredictionHistoryResponse {
  items: StoredPrediction[]
}
