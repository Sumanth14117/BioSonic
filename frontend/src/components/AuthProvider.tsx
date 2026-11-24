import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { changeAccountPassword, getAuthState, loginAccount, logoutAccount, registerAccount, updateAccountEmail } from '../lib/api'
import { AuthContext, type AuthContextValue } from './AuthContext'
import type { AuthUser } from '../types/api'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [csrfToken, setCsrfToken] = useState('')
  const [loading, setLoading] = useState(true)
  const [authError, setAuthError] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const state = await getAuthState()
      setUser(state.user)
      setCsrfToken(state.csrf_token)
      setAuthError('')
    } catch {
      setUser(null)
      setCsrfToken('')
      setAuthError('BioSonic could not reach the authentication service.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    getAuthState().then((state) => {
      if (!active) return
      setUser(state.user)
      setCsrfToken(state.csrf_token)
      setAuthError('')
    }).catch(() => {
      if (active) {
        setUser(null)
        setCsrfToken('')
        setAuthError('BioSonic could not reach the authentication service.')
      }
    }).finally(() => { if (active) setLoading(false) })

    const onUnauthorized = () => {
      setUser(null)
      setCsrfToken('')
      setAuthError('Your session expired. Refresh the sign-in session to continue.')
    }
    window.addEventListener('biosonic:unauthorized', onUnauthorized)
    return () => {
      active = false
      window.removeEventListener('biosonic:unauthorized', onUnauthorized)
    }
  }, [])

  const value = useMemo<AuthContextValue>(() => ({
    user,
    csrfToken,
    loading,
    authError,
    refresh,
    async register(email, password) {
      try {
        const state = await registerAccount(email, password, csrfToken)
        setUser(state.user)
        setCsrfToken(state.csrf_token)
        setAuthError('')
      } catch (error) {
        const state = await getAuthState().catch(() => null)
        if (state) { setUser(state.user); setCsrfToken(state.csrf_token) }
        else setAuthError('BioSonic could not reach the authentication service.')
        throw error
      }
    },
    async login(email, password) {
      try {
        const state = await loginAccount(email, password, csrfToken)
        setUser(state.user)
        setCsrfToken(state.csrf_token)
        setAuthError('')
      } catch (error) {
        const state = await getAuthState().catch(() => null)
        if (state) { setUser(state.user); setCsrfToken(state.csrf_token) }
        else setAuthError('BioSonic could not reach the authentication service.')
        throw error
      }
    },
    async logout() {
      await logoutAccount(csrfToken)
      setUser(null)
      const state = await getAuthState()
      setCsrfToken(state.csrf_token)
    },
    async updateEmail(email) {
      setUser(await updateAccountEmail(email, csrfToken))
    },
    async changePassword(currentPassword, newPassword) {
      await changeAccountPassword(currentPassword, newPassword, csrfToken)
      setUser(null)
      const state = await getAuthState()
      setCsrfToken(state.csrf_token)
    },
  }), [user, csrfToken, loading, authError, refresh])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
