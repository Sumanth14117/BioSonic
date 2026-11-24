import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { StoredPrediction } from '../types/api'

const api = vi.hoisted(() => ({ getPrediction: vi.fn() }))
vi.mock('../lib/api', () => ({ getPrediction: api.getPrediction }))

import { ResultsPage } from './ResultsPage'

const saved: StoredPrediction = {
  id: 'record-1', file_name: 'evening-field.ogg', file_size: 14_200, created_at: 1_735_776_000,
  response: {
    prediction_id: 'record-1', model: 'biosonic-m4-cnn', model_version: '95c770685484', prediction: 'burowl', confidence: 0.74,
    probabilities: { compot1: 0.02, rufhor2: 0.03, saytan1: 0.04, crbthr1: 0.05, rutjac1: 0.06, rufnig1: 0.02, chobla1: 0.04, burowl: 0.74 },
    audio: { duration_seconds: 12.5, sample_rate: 32_000, channels: 1, sample_count: 400_000, format: 'OGG/VORBIS' },
  },
}

function renderResults() {
  return render(<MemoryRouter initialEntries={['/results/record-1']}><Routes><Route path="/results/:id" element={<ResultsPage />} /></Routes></MemoryRouter>)
}

beforeEach(() => api.getPrediction.mockResolvedValue(saved))

describe('prediction results', () => {
  it('renders actual prediction, probability visualization, model and recording metadata', async () => {
    renderResults()
    expect(await screen.findByRole('heading', { name: 'burowl' })).toBeTruthy()
    expect(screen.getByRole('img', { name: 'Model confidence 74 percent' })).toBeTruthy()
    expect(screen.getByRole('img', { name: "Ranked probability bars for the model's trained classes" })).toBeTruthy()
    expect(screen.getByText('biosonic-m4-cnn')).toBeTruthy()
    expect(screen.getByText('95c770685484')).toBeTruthy()
    expect(screen.getByText('evening-field.ogg')).toBeTruthy()
    expect(screen.getByText('12.50s')).toBeTruthy()
    expect(screen.getByText(/not calibrated probabilities of biological presence/)).toBeTruthy()
  })

  it('shows an owner-scoped unavailable state when the API cannot return a record', async () => {
    api.getPrediction.mockRejectedValueOnce(new Error('The request could not be completed.'))
    renderResults()
    expect(await screen.findByText('This prediction is not available to this account')).toBeTruthy()
  })
})
