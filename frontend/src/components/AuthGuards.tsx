import type { ReactNode } from 'react'
import { Navigate, Outlet } from 'react-router'
import { useAuth } from './AuthContext'

function SessionLoading() {
  return <main className="session-loading" role="status"><span className="spinner" /> Checking your local session…</main>
}

export function RequireAuth() {
  const { user, loading } = useAuth()
  if (loading) return <SessionLoading />
  return user ? <Outlet /> : <Navigate to="/login" replace />
}

export function PublicOnly({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <SessionLoading />
  return user ? <Navigate to="/" replace /> : children
}
