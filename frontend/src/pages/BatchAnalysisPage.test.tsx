import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import { AuthContext, type AuthContextValue } from '../components/AuthContext'
import { BatchAnalysisPage } from './BatchAnalysisPage'

const api = vi.hoisted(() => ({ requestPrediction: vi.fn() }))
vi.mock('../lib/api', async (importOriginal) => ({ ...(await importOriginal<typeof import('../lib/api')>()), requestPrediction: api.requestPrediction }))

const auth: AuthContextValue = { user: { id: 'u-1', email: 'field@example.org' }, csrfToken: 'csrf', loading: false, authError: '', refresh: async () => undefined, register: async () => undefined, login: async () => undefined, logout: async () => undefined, updateEmail: async () => undefined, changePassword: async () => undefined }
function renderBatch() { return render(<AuthContext.Provider value={auth}><MemoryRouter><BatchAnalysisPage /></MemoryRouter></AuthContext.Provider>) }
function response(id: string) { return { model: 'biosonic-m4-cnn', model_version: 'version', prediction_id: id, prediction: 'banana', confidence: 0.75, probabilities: { banana: 0.75, other: 0.25 }, audio: { duration_seconds: 5, sample_rate: 32000, channels: 1, sample_count: 160000, format: 'OGG/VORBIS' } } }

describe('batch analysis queue', () => {
  it('validates each added file and reports duplicate selections', () => {
    renderBatch()
    const input = screen.getByLabelText('Choose multiple OGG audio files')
    const valid = new File(['audio'], 'one.ogg', { lastModified: 1 })
    fireEvent.change(input, { target: { files: [valid, new File(['x'], 'bad.wav'), valid] } })
    expect(screen.getByText('one.ogg')).toBeTruthy()
    expect(screen.getByRole('alert').textContent).toMatch(/Only \.ogg/)
    expect(screen.getByRole('alert').textContent).toMatch(/duplicate selection/i)
    expect(api.requestPrediction).not.toHaveBeenCalled()
  })

  it('uses actual API response for completed status and offers per-recording result', async () => {
    api.requestPrediction.mockResolvedValueOnce(response('prediction-id'))
    renderBatch()
    fireEvent.change(screen.getByLabelText('Choose multiple OGG audio files'), { target: { files: [new File(['audio'], 'one.ogg')] } })
    fireEvent.click(screen.getByRole('button', { name: /Start queue/ }))
    await waitFor(() => expect(api.requestPrediction).toHaveBeenCalledWith(expect.any(File), 'csrf', expect.any(Function), expect.any(AbortSignal)))
    expect(await screen.findByText(/banana · 75% score/)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'View result for one.ogg' }).getAttribute('href')).toBe('/results/prediction-id')
    expect(screen.getByText('Scores')).toBeTruthy()
  })

  it('isolates failures and retries only the failed recording', async () => {
    api.requestPrediction.mockRejectedValueOnce(new Error('temporary failure')).mockResolvedValueOnce(response('retry-id'))
    renderBatch()
    fireEvent.change(screen.getByLabelText('Choose multiple OGG audio files'), { target: { files: [new File(['audio'], 'retry.ogg')] } })
    fireEvent.click(screen.getByRole('button', { name: /Start queue/ }))
    expect(await screen.findByText('failed')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry retry.ogg' }))
    expect(await screen.findByText(/banana · 75% score/)).toBeTruthy()
    expect(api.requestPrediction).toHaveBeenCalledTimes(2)
  })

  it('aborts active requests and reports them as cancelled', async () => {
    api.requestPrediction.mockImplementationOnce((_file, _csrf, _progress, signal: AbortSignal) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true })))
    renderBatch()
    fireEvent.change(screen.getByLabelText('Choose multiple OGG audio files'), { target: { files: [new File(['audio'], 'cancel.ogg')] } })
    fireEvent.click(screen.getByRole('button', { name: /Start queue/ }))
    await waitFor(() => expect(api.requestPrediction).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: /Cancel requests/ }))
    expect(await screen.findByText('cancelled')).toBeTruthy()
  })

  it('keeps batch request concurrency at two', async () => {
    let active = 0, maximum = 0
    api.requestPrediction.mockImplementation(async (file: File) => {
      active++
      maximum = Math.max(maximum, active)
      await new Promise((resolve) => setTimeout(resolve, 10))
      active--
      return response(file.name)
    })
    renderBatch()
    fireEvent.change(screen.getByLabelText('Choose multiple OGG audio files'), { target: { files: [new File(['1'], 'one.ogg'), new File(['2'], 'two.ogg'), new File(['3'], 'three.ogg')] } })
    fireEvent.click(screen.getByRole('button', { name: /Start queue/ }))
    await waitFor(() => expect(screen.getAllByText('completed')).toHaveLength(3))
    expect(maximum).toBe(2)
  })
})
