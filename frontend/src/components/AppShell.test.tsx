import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import { AppShell } from './AppShell'
import { AuthContext } from './AuthContext'

describe('primary navigation', () => {
  it('navigates between the overview and model registry', async () => {
    render(
      <AuthContext.Provider value={{ user: { id: 'test-user', email: 'listener@example.org' }, csrfToken: 'csrf', loading: false, authError: '', refresh: async () => undefined, register: async () => undefined, login: async () => undefined, logout: async () => undefined, updateEmail: async () => undefined, changePassword: async () => undefined }}>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<h1>Overview fixture</h1>} />
            <Route path="models" element={<h1>Model registry fixture</h1>} />
          </Route>
        </Routes>
      </MemoryRouter>
      </AuthContext.Provider>,
    )
    expect(screen.getByText('Overview fixture')).toBeTruthy()
    fireEvent.click(screen.getByRole('link', { name: /Model information/ }))
    expect(await screen.findByText('Model registry fixture')).toBeTruthy()
  })

  it('shows the signed-in account menu and calls logout', async () => {
    const logout = vi.fn(async () => undefined)
    render(
      <AuthContext.Provider value={{ user: { id: 'test-user', email: 'listener@example.org' }, csrfToken: 'csrf', loading: false, authError: '', refresh: async () => undefined, register: async () => undefined, login: async () => undefined, logout, updateEmail: async () => undefined, changePassword: async () => undefined }}>
        <MemoryRouter initialEntries={['/']}>
          <Routes>
            <Route element={<AppShell />}><Route index element={<h1>Overview fixture</h1>} /><Route path="login" element={<h1>Sign-in fixture</h1>} /></Route>
          </Routes>
        </MemoryRouter>
      </AuthContext.Provider>,
    )
    fireEvent.click(screen.getByLabelText('Account menu for listener@example.org'))
    fireEvent.click(screen.getByRole('button', { name: 'Sign out' }))
    await waitFor(() => expect(logout).toHaveBeenCalledTimes(1))
    expect(await screen.findByRole('heading', { name: 'Sign-in fixture' })).toBeTruthy()
  })
})
