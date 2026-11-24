import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getApiError, requestPrediction, validateUploadFile } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('upload validation', () => {
  it('accepts non-empty OGG files below the configured limit', () => {
    expect(validateUploadFile(new File(['audio'], 'recording.ogg', { type: 'audio/ogg' }))).toBeNull()
  })

  it('rejects unsupported extensions, empty files, and oversized files', () => {
    expect(validateUploadFile(new File(['x'], 'recording.wav'))).toMatch(/\.ogg/)
    expect(validateUploadFile(new File([], 'empty.ogg'))).toMatch(/empty/i)
    const oversized = new File([new Uint8Array(25 * 1024 * 1024 + 1)], 'large.ogg')
    expect(validateUploadFile(oversized)).toMatch(/25 MiB/)
  })
})

describe('API error handling', () => {
  it('maps the backend error envelope without exposing internal details', () => {
    const error = getApiError(422, { detail: { code: 'unsupported_sample_rate', message: 'Audio must use 32 kHz.' } })
    expect(error).toBeInstanceOf(ApiError)
    expect(error.code).toBe('unsupported_sample_rate')
    expect(error.message).toBe('Audio must use 32 kHz.')
  })

  it('provides a safe fallback for unreadable server errors', () => {
    const error = getApiError(500, '<html>internal traceback</html>')
    expect(error.message).not.toContain('traceback')
    expect(error.status).toBe(500)
  })
})

describe('prediction upload client', () => {
  it('sends multipart field `file`, reports progress, and parses the typed result', async () => {
    const progress = vi.fn()
    let sentBody: FormData | undefined
    let csrfHeader = ''

    class MockXhr {
      upload = { onprogress: null as ((event: ProgressEvent) => void) | null }
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      ontimeout: (() => void) | null = null
      onabort: (() => void) | null = null
      responseText = ''
      responseType = ''
      timeout = 0
      status = 0
      headers: Record<string, string> = {}
      open = vi.fn()
      setRequestHeader = (key: string, value: string) => { this.headers[key] = value; if (key === 'X-CSRF-Token') csrfHeader = value }
      send = (body: FormData) => {
        sentBody = body
        this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 } as ProgressEvent)
        this.status = 200
        this.responseText = JSON.stringify({ model: 'biosonic-m4-cnn', model_version: 'abc123', prediction: 'burowl', confidence: 0.8, probabilities: { burowl: 0.8, other: 0.2 }, audio: { duration_seconds: 5, sample_rate: 32000, channels: 1, sample_count: 160000, format: 'OGG/VORBIS' } })
        this.onload?.()
      }
    }
    vi.stubGlobal('XMLHttpRequest', MockXhr)

    const file = new File(['ogg bytes'], 'bird.ogg', { type: 'audio/ogg' })
    const result = await requestPrediction(file, 'csrf-test-token', progress)

    const submitted = (sentBody as FormData).get('file') as File
    expect(submitted.name).toBe(file.name)
    expect(submitted.size).toBe(file.size)
    expect(progress).toHaveBeenCalledWith(50)
    expect(csrfHeader).toBe('csrf-test-token')
    expect(result.model).toBe('biosonic-m4-cnn')
    expect(result.prediction).toBe('burowl')
  })
})
