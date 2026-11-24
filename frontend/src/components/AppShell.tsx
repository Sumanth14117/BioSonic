import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router'
import { Activity, AudioLines, BookOpen, ChevronDown, CircleHelp, Clock3, FlaskConical, Layers3, Leaf, LogOut, Menu, UserRound, Waves, X } from 'lucide-react'
import { useAuth } from './AuthContext'
import { getReadiness } from '../lib/api'
import { ToastProvider } from './ToastProvider'
import { useToast } from '../lib/toastContext'

const navigation = [
  { to: '/', label: 'Overview', icon: Activity, end: true },
  { to: '/analysis', label: 'Audio analysis', icon: AudioLines },
  { to: '/batch', label: 'Batch analysis', icon: Layers3 },
  { to: '/history', label: 'Prediction history', icon: Clock3 },
  { to: '/models', label: 'Model information', icon: FlaskConical },
  { to: '/about', label: 'About BioSonic', icon: BookOpen },
]

const pageTitles: Record<string, string> = {
  '/': 'Overview',
  '/analysis': 'Audio analysis',
  '/batch': 'Batch analysis',
  '/history': 'Prediction history',
  '/account': 'Account settings',
  '/results': 'Prediction results',
  '/models': 'Model information',
  '/about': 'About BioSonic',
}

function AccountMenu() {
  const { user, logout } = useAuth()
  const { showToast } = useToast()
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  if (!user) return null
  async function signOut() {
    setBusy(true)
    try { await logout(); navigate('/login', { replace: true }) } catch { showToast('Sign out could not be confirmed. Refresh and try again.', 'error') } finally { setBusy(false) }
  }
  return <details className="account-menu">
    <summary aria-label={`Account menu for ${user.email}`}><span className="account-avatar"><UserRound size={15} /></span><span className="account-menu-email">{user.email}</span><ChevronDown size={14} /></summary>
    <div className="account-menu-popover"><div className="account-menu-identity"><small>SIGNED IN AS</small><strong>{user.email}</strong></div><NavLink to="/account"><UserRound size={15} /> Account settings</NavLink><button type="button" disabled={busy} onClick={() => void signOut()}><LogOut size={15} /> {busy ? 'Signing out…' : 'Sign out'}</button></div>
  </details>
}

function BackendIndicator() {
  const [status, setStatus] = useState<'checking' | 'connected' | 'unavailable'>('checking')

  useEffect(() => {
    let active = true
    getReadiness().then((response) => {
      if (active) setStatus(response.status === 'ready' ? 'connected' : 'unavailable')
    }).catch(() => {
      if (active) setStatus('unavailable')
    })
    return () => { active = false }
  }, [])

  const label = status === 'connected' ? 'API connected' : status === 'checking' ? 'Checking API' : 'API offline'
  return <span className={`api-indicator api-${status}`}><span className="status-dot" />{label}</span>
}

export function AppShell() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const location = useLocation()
  const pageTitle = location.pathname.startsWith('/results/') ? 'Prediction results' : (pageTitles[location.pathname] ?? 'BioSonic')

  useEffect(() => {
    document.title = `${pageTitle} · BioSonic`
  }, [pageTitle, location.pathname])

  const links = navigation.map(({ to, label, icon: Icon, end }) => (
    <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`} onClick={() => setMobileNavOpen(false)}>
      <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
      <span>{label}</span>
    </NavLink>
  ))

  return (
    <ToastProvider>
      <div className="app-frame">
        <aside className={`sidebar${mobileNavOpen ? ' sidebar-open' : ''}`} aria-label="Primary navigation">
          <NavLink to="/" className="brand-lockup" aria-label="BioSonic overview">
            <span className="brand-mark"><Waves size={21} strokeWidth={2.2} /></span>
            <span className="brand-wordmark">bio<span>sonic</span><small>FIELD INTELLIGENCE</small></span>
          </NavLink>
          <div className="workspace-label">WORKSPACE</div>
          <nav className="primary-nav">{links}</nav>
          <div className="sidebar-spacer" />
          <div className="sidebar-note">
            <span className="sidebar-note-icon"><Leaf size={16} /></span>
            <p><strong>Research preview</strong><br />Eight-class audio classification. Not a wildlife census.</p>
          </div>
          <div className="sidebar-footer"><span className="version-dot" /> M4 model · local inference</div>
        </aside>

        {mobileNavOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}

        <div className="main-column">
          <header className="topbar">
            <button className="mobile-menu button-icon" aria-label={mobileNavOpen ? 'Close navigation' : 'Open navigation'} onClick={() => setMobileNavOpen((open) => !open)}>
              {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="breadcrumb"><span>BioSonic</span><span className="breadcrumb-divider">/</span><strong>{pageTitle}</strong></div>
            <div className="topbar-right"><span className="environment-label"><span className="environment-mark" /> LOCAL ENVIRONMENT</span><BackendIndicator /><AccountMenu /></div>
          </header>
          <main id="main-content" className="page-content"><Outlet /></main>
          <footer className="app-footer"><span>BioSonic · Wildlife audio intelligence</span><span><CircleHelp size={13} /> Research tool · outputs require expert review</span></footer>
        </div>
      </div>
    </ToastProvider>
  )
}
