import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import { AuthContext, type AuthContextValue } from '../components/AuthContext'
import { ToastProvider } from '../components/ToastProvider'

const api = vi.hoisted(() => ({ requestPrediction: vi.fn() }))
vi.mock('../lib/api', async (importOriginal) => ({ ...(await importOriginal<typeof import('../lib/api')>()), requestPrediction: api.requestPrediction }))
vi.mock('../components/AudioIntelligence', () => ({ AudioIntelligence: () => <div><audio aria-label="Preview selected audio" /><div>Audio intelligence fixture</div></div> }))

import { AnalysisPage } from './AnalysisPage'

function renderAnalysis() {
  const auth: AuthContextValue = { user: { id: 'u-1', email: 'field@example.org' }, csrfToken: 'account-csrf', loading: false, authError: '', refresh: async () => undefined, register: async () => undefined, login: async () => undefined, logout: async () => undefined, updateEmail: async () => undefined, changePassword: async () => undefined }
  return render(<AuthContext.Provider value={auth}><ToastProvider><MemoryRouter initialEntries={['/analysis']}><Routes><Route path="/analysis" element={<AnalysisPage />} /><Route path="/results/:id" element={<h1>Saved result destination</h1>} /></Routes></MemoryRouter></ToastProvider></AuthContext.Provider>)
}

describe('audio analysis workspace', () => {
  it('rejects unsupported files and preserves the upload contract', () => {
    renderAnalysis()
    fireEvent.change(screen.getByLabelText('Choose OGG audio file'), { target: { files: [new File(['audio'], 'recording.wav')] } })
    expect(screen.getByRole('alert').textContent).toMatch(/Only \.ogg/)
    expect(screen.getByText(/Mono · 32,000 Hz/)).toBeTruthy()
    expect(api.requestPrediction).not.toHaveBeenCalled()
  })

  it('previews an accepted recording and navigates after the real API response', async () => {
    let reportProgress!: (percentage: number) => void
    let settlePrediction!: (result: unknown) => void
    api.requestPrediction.mockImplementationOnce((_file, _csrfToken, onProgress) => new Promise((resolve) => {
      reportProgress = onProgress
      settlePrediction = resolve
    }))
    renderAnalysis()
    const file = new File(['ogg-data'], 'forest.ogg', { type: 'audio/ogg' })
    fireEvent.change(screen.getByLabelText('Choose OGG audio file'), { target: { files: [file] } })
    expect(await screen.findByText('forest.ogg')).toBeTruthy()
    expect(screen.getByLabelText('Preview selected audio')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Analyse audio/ }))
    expect(await screen.findByText('Preparing recording upload')).toBeTruthy()
    await waitFor(() => expect(api.requestPrediction).toHaveBeenCalled())
    act(() => reportProgress(43))
    expect(screen.getByText('Uploading recording · 43%')).toBeTruthy()
    act(() => reportProgress(100))
    expect(screen.getByText('Upload complete · waiting for prediction')).toBeTruthy()
    expect(screen.queryByText('100%')).toBeNull()
    await act(async () => settlePrediction({ prediction_id: 'prediction-42', model: 'biosonic-m4-cnn' }))
    expect(await screen.findByRole('heading', { name: 'Saved result destination' })).toBeTruthy()
    expect(api.requestPrediction).toHaveBeenCalledWith(file, 'account-csrf', expect.any(Function))
  })
})
