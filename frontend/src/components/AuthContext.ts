import { createContext, useContext } from 'react'
import type { AuthUser } from '../types/api'

export interface AuthContextValue {
  user: AuthUser | null
  csrfToken: string
  loading: boolean
  authError: string
  refresh: () => Promise<void>
  register: (email: string, password: string) => Promise<void>
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  updateEmail: (email: string) => Promise<void>
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}
