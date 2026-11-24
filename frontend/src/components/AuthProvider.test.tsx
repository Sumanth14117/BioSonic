import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  changeAccountPassword: vi.fn(),
  getAuthState: vi.fn(),
  loginAccount: vi.fn(),
  logoutAccount: vi.fn(),
  registerAccount: vi.fn(),
  updateAccountEmail: vi.fn(),
}))

vi.mock('../lib/api', () => api)

import { useAuth } from './AuthContext'
import { AuthProvider } from './AuthProvider'

function Probe() {
  const auth = useAuth()
  return <div>
    <output aria-label="account">{auth.user?.email ?? 'anonymous'}</output>
    <output aria-label="csrf">{auth.csrfToken}</output>
    <button onClick={() => void auth.register('new@example.org', 'a secure passphrase 42')}>register</button>
    <button onClick={() => void auth.login('new@example.org', 'a secure passphrase 42')}>login</button>
    <button onClick={() => void auth.logout()}>logout</button>
  </div>
}

const signedIn = { user: { id: 'user-1', email: 'new@example.org' }, csrf_token: 'session-csrf' }

afterEach(() => vi.resetAllMocks())

describe('authentication provider', () => {
  it('restores an anonymous session and updates state after registration', async () => {
    api.getAuthState.mockResolvedValueOnce({ user: null, csrf_token: 'anonymous-csrf' })
    api.registerAccount.mockResolvedValueOnce(signedIn)
    render(<AuthProvider><Probe /></AuthProvider>)

    expect((await screen.findByLabelText('account')).textContent).toBe('anonymous')
    expect(screen.getByLabelText('csrf').textContent).toBe('anonymous-csrf')
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'register' })) })

    expect(api.registerAccount).toHaveBeenCalledWith('new@example.org', 'a secure passphrase 42', 'anonymous-csrf')
    expect(screen.getByLabelText('account').textContent).toBe('new@example.org')
    expect(screen.getByLabelText('csrf').textContent).toBe('session-csrf')
  })

  it('refreshes the CSRF challenge after login and clears the user after logout', async () => {
    api.getAuthState
      .mockResolvedValueOnce({ user: null, csrf_token: 'anonymous-csrf' })
      .mockResolvedValueOnce({ user: null, csrf_token: 'fresh-anonymous-csrf' })
    api.loginAccount.mockResolvedValueOnce(signedIn)
    api.logoutAccount.mockResolvedValueOnce(undefined)
    render(<AuthProvider><Probe /></AuthProvider>)

    await screen.findByLabelText('csrf')
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'login' })) })
    expect(api.loginAccount).toHaveBeenCalledWith('new@example.org', 'a secure passphrase 42', 'anonymous-csrf')
    expect(screen.getByLabelText('account').textContent).toBe('new@example.org')

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'logout' })) })
    await waitFor(() => expect(screen.getByLabelText('account').textContent).toBe('anonymous'))
    expect(api.logoutAccount).toHaveBeenCalledWith('session-csrf')
    expect(screen.getByLabelText('csrf').textContent).toBe('fresh-anonymous-csrf')
  })
})
