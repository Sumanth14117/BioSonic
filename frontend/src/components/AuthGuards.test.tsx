import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { describe, expect, it } from 'vitest'
import { AuthContext, type AuthContextValue } from './AuthContext'
import { PublicOnly, RequireAuth } from './AuthGuards'

function authValue(overrides: Partial<AuthContextValue> = {}): AuthContextValue {
  return {
    user: null, csrfToken: 'csrf-token', loading: false, authError: '',
    refresh: async () => undefined, register: async () => undefined, login: async () => undefined,
    logout: async () => undefined, updateEmail: async () => undefined, changePassword: async () => undefined,
    ...overrides,
  }
}

function renderRoutes(auth: AuthContextValue, initial = '/private') {
  return render(<AuthContext.Provider value={auth}><MemoryRouter initialEntries={[initial]}><Routes>
    <Route path="/login" element={<h1>Sign-in page</h1>} />
    <Route path="/" element={<h1>Overview page</h1>} />
    <Route path="/private" element={<RequireAuth />}><Route index element={<h1>Private history</h1>} /></Route>
    <Route path="/register" element={<PublicOnly><h1>Registration page</h1></PublicOnly>} />
  </Routes></MemoryRouter></AuthContext.Provider>)
}

describe('authentication route guards', () => {
  it('redirects unauthenticated users away from private pages', () => {
    renderRoutes(authValue())
    expect(screen.getByRole('heading', { name: 'Sign-in page' })).toBeTruthy()
  })

  it('allows authenticated users to reach private pages', () => {
    renderRoutes(authValue({ user: { id: 'user-1', email: 'listener@example.org' } }))
    expect(screen.getByRole('heading', { name: 'Private history' })).toBeTruthy()
  })

  it('redirects an authenticated user away from registration', () => {
    renderRoutes(authValue({ user: { id: 'user-1', email: 'listener@example.org' } }), '/register')
    expect(screen.getByRole('heading', { name: 'Overview page' })).toBeTruthy()
  })

  it('waits for session restoration before redirecting', () => {
    renderRoutes(authValue({ loading: true }))
    expect(screen.getByRole('status').textContent).toContain('Checking your local session')
  })
})
