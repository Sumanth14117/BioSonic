import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AuthContext, type AuthContextValue } from '../components/AuthContext'
import type { ApiModel, StoredPrediction } from '../types/api'

const api = vi.hoisted(() => ({ getModels: vi.fn(), getPredictionHistory: vi.fn(), getReadiness: vi.fn() }))
vi.mock('../lib/api', () => api)

import { DashboardPage } from './DashboardPage'

const model: ApiModel = { id: 'biosonic-m4-cnn', task: 'Recording-level classification', status: 'ready', version: 'abc123', classes: ['burowl', 'chobla1'], limitations: [], device: 'cuda:0' }
const record: StoredPrediction = {
  id: 'p-1', file_name: 'forest.ogg', file_size: 1000, created_at: 1_735_776_000,
  response: { prediction_id: 'p-1', model: model.id, model_version: 'abc123', prediction: 'burowl', confidence: 0.74, probabilities: { burowl: 0.74, chobla1: 0.26 }, audio: { duration_seconds: 5, sample_rate: 32_000, channels: 1, sample_count: 160_000, format: 'OGG/VORBIS' } },
}

function renderDashboard() {
  const auth: AuthContextValue = { user: { id: 'u-1', email: 'sumanth@example.org' }, csrfToken: 'csrf', loading: false, authError: '', refresh: async () => undefined, register: async () => undefined, login: async () => undefined, logout: async () => undefined, updateEmail: async () => undefined, changePassword: async () => undefined }
  return render(<AuthContext.Provider value={auth}><MemoryRouter><DashboardPage /></MemoryRouter></AuthContext.Provider>)
}

beforeEach(() => {
  api.getReadiness.mockResolvedValue({ status: 'ready', models: {} })
  api.getModels.mockResolvedValue({ models: [model] })
  api.getPredictionHistory.mockResolvedValue({ items: [record, { ...record, id: 'p-2', file_name: 'canopy.ogg', created_at: record.created_at + 1, response: { ...record.response, prediction_id: 'p-2', confidence: 0.81 } }] })
})

describe('dashboard history analytics', () => {
  it('shows account-derived totals, top classes, model data and recent files', async () => {
    renderDashboard()
    expect(await screen.findByRole('heading', { name: 'Good to see you, Sumanth.' })).toBeTruthy()
    expect(await screen.findByText('2 of 2 saved predictions')).toBeTruthy()
    expect(screen.getByText('5.0 s')).toBeTruthy()
    expect(screen.getByText('cuda:0')).toBeTruthy()
    expect(screen.getByText('forest.ogg')).toBeTruthy()
    expect(screen.getByText('canopy.ogg')).toBeTruthy()
    await waitFor(() => expect(api.getPredictionHistory).toHaveBeenCalledTimes(1))
  })

  it('shows real empty states when the account has no predictions', async () => {
    api.getPredictionHistory.mockResolvedValueOnce({ items: [] })
    renderDashboard()
    expect(await screen.findByText('No activity yet')).toBeTruthy()
    expect(screen.getByText('No class counts yet')).toBeTruthy()
    expect(screen.getByText('No duration data yet')).toBeTruthy()
    expect(screen.getByText('No saved predictions yet')).toBeTruthy()
  })
})
