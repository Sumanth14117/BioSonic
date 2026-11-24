import type { ApiErrorPayload, ApiReadiness, AuthStateResponse, AuthUser, ModelsResponse, PredictionHistoryResponse, PredictionResponse, StoredPrediction } from '../types/api'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const REQUEST_TIMEOUT_MS = 120_000

export class ApiError extends Error {
  public readonly status: number
  public readonly code: string

  constructor(
    message: string,
    status: number,
    code: string,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function endpoint(path: string): string {
  return `${API_BASE_URL}${path}`
}

function notifyUnauthorized(status: number) {
  if (status === 401 && typeof window !== 'undefined') window.dispatchEvent(new Event('biosonic:unauthorized'))
}

export function getApiError(status: number, payload: unknown): ApiError {
  const body = payload as ApiErrorPayload | null
  const detail = body && typeof body === 'object' ? body.detail : undefined
  if (detail && typeof detail === 'object') {
    return new ApiError(detail.message || 'The API request could not be completed.', status, detail.code || 'request_failed')
  }

  const messages: Record<number, [string, string]> = {
    400: ['invalid_upload', 'The selected filename or upload is invalid.'],
    413: ['upload_too_large', 'The audio file exceeds the 25 MiB upload limit.'],
    415: ['unsupported_format', 'Upload a valid OGG/Vorbis audio file.'],
    422: ['audio_validation_failed', 'Audio must be decodable mono 32 kHz OGG/Vorbis between 1 and 180 seconds.'],
    500: ['inference_failed', 'The model could not analyze this recording. Try again.'],
    503: ['model_unavailable', 'The M4 model is not ready. Check the backend status.'],
  }
  const [code, message] = messages[status] ?? ['request_failed', 'The API request could not be completed.']
  return new ApiError(message, status, code)
}

async function getJson<T>(path: string): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 15_000)
  try {
    const response = await fetch(endpoint(path), {
      method: 'GET',
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: controller.signal,
    })
    const payload: unknown = await response.json().catch(() => null)
    if (!response.ok) {
      notifyUnauthorized(response.status)
      throw getApiError(response.status, payload)
    }
    return payload as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('The backend did not respond in time.', 0, 'timeout')
    }
    throw new ApiError('BioSonic could not reach the backend. Confirm that the API is running.', 0, 'backend_unavailable')
  } finally {
    window.clearTimeout(timeout)
  }
}

export function getReadiness(): Promise<ApiReadiness> {
  return getJson<ApiReadiness>('/api/v1/health')
}

export function getModels(): Promise<ModelsResponse> {
  return getJson<ModelsResponse>('/api/v1/models')
}

async function postJson<T>(path: string, body: unknown, csrfToken: string): Promise<T> {
  const response = await fetch(endpoint(path), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify(body),
    cache: 'no-store',
  })
  const payload: unknown = response.status === 204 ? undefined : await response.json().catch(() => null)
  if (!response.ok) {
    throw getApiError(response.status, payload)
  }
  return payload as T
}

export function getAuthState(): Promise<AuthStateResponse> {
  return getJson<AuthStateResponse>('/api/v1/auth/csrf')
}

export function registerAccount(email: string, password: string, csrfToken: string): Promise<AuthStateResponse> {
  return postJson<AuthStateResponse>('/api/v1/auth/register', { email, password }, csrfToken)
}

export function loginAccount(email: string, password: string, csrfToken: string): Promise<AuthStateResponse> {
  return postJson<AuthStateResponse>('/api/v1/auth/login', { email, password }, csrfToken)
}

export async function logoutAccount(csrfToken: string): Promise<void> {
  await postJson<void>('/api/v1/auth/logout', {}, csrfToken)
}

export function updateAccountEmail(email: string, csrfToken: string): Promise<AuthUser> {
  return fetch(endpoint('/api/v1/auth/me'), {
    method: 'PATCH',
    credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify({ email }),
  }).then(async (response) => {
    const payload: unknown = await response.json().catch(() => null)
    if (!response.ok) {
      notifyUnauthorized(response.status)
      throw getApiError(response.status, payload)
    }
    return payload as AuthUser
  })
}

export async function changeAccountPassword(currentPassword: string, newPassword: string, csrfToken: string): Promise<void> {
  const response = await fetch(endpoint('/api/v1/auth/me/password'), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
  const payload: unknown = response.status === 204 ? undefined : await response.json().catch(() => null)
  if (!response.ok) {
    notifyUnauthorized(response.status)
    throw getApiError(response.status, payload)
  }
}

export function getPredictionHistory(): Promise<PredictionHistoryResponse> {
  return getJson<PredictionHistoryResponse>('/api/v1/history')
}

export function getPrediction(predictionId: string): Promise<StoredPrediction> {
  return getJson<StoredPrediction>(`/api/v1/history/${encodeURIComponent(predictionId)}`)
}

export function requestPrediction(
  file: File,
  csrfToken: string,
  onProgress?: (percentage: number) => void,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    if (signal?.aborted) {
      reject(new ApiError('The audio analysis request was cancelled.', 0, 'aborted'))
      return
    }
    xhr.open('POST', endpoint('/api/v1/predict'))
    xhr.withCredentials = true
    xhr.timeout = REQUEST_TIMEOUT_MS
    xhr.setRequestHeader('Accept', 'application/json')
    xhr.setRequestHeader('X-CSRF-Token', csrfToken)
    xhr.responseType = 'text'

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(Math.min(100, Math.round((event.loaded / event.total) * 100)))
    }

    xhr.onload = () => {
      let payload: unknown = null
      try {
        payload = JSON.parse(xhr.responseText)
      } catch {
        payload = null
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload as PredictionResponse)
      } else {
        notifyUnauthorized(xhr.status)
        reject(getApiError(xhr.status, payload))
      }
    }

    xhr.onerror = () => reject(new ApiError('BioSonic could not reach the backend. Confirm that the API is running.', 0, 'backend_unavailable'))
    xhr.ontimeout = () => reject(new ApiError('Audio analysis timed out. Try a shorter recording.', 0, 'timeout'))
    xhr.onabort = () => reject(new ApiError('The audio analysis request was cancelled.', 0, 'aborted'))
    const abortRequest = () => xhr.abort()
    signal?.addEventListener('abort', abortRequest, { once: true })
    xhr.onloadend = () => signal?.removeEventListener('abort', abortRequest)

    const form = new FormData()
    form.append('file', file, file.name)
    xhr.send(form)
  })
}

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024

export function validateUploadFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith('.ogg')) return 'Only .ogg audio files are accepted.'
  if (file.size === 0) return 'This file is empty. Choose an audio recording.'
  if (file.size > MAX_UPLOAD_BYTES) return 'This file exceeds the 25 MiB upload limit.'
  return null
}
