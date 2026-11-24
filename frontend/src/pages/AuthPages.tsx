import { useState, type FormEvent, type ReactNode } from 'react'
import { AudioLines, Eye, EyeOff, Leaf, LockKeyhole, Mail, ShieldCheck, Waves } from 'lucide-react'
import { Link, Navigate, useNavigate } from 'react-router'
import { useAuth } from '../components/AuthContext'
import { ApiError } from '../lib/api'

function AuthFrame({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <main className="auth-page">
    <section className="auth-story" aria-label="BioSonic introduction">
      <Link to="/login" className="auth-brand"><span className="auth-brand-mark"><Waves size={21} /></span><span>bio<span>sonic</span><small>FIELD INTELLIGENCE</small></span></Link>
      <div className="auth-story-copy"><span className="eyebrow"><span className="live-mark" /> WILDLIFE AUDIO INTELLIGENCE</span><h1>Listen closer.<br /><em>Understand more.</em></h1><p>A private research workspace for exploring wildlife recordings with transparent, reviewable model outputs.</p><div className="auth-trust"><span><ShieldCheck size={16} /> Private local sessions</span><span><Leaf size={16} /> Research, with context</span></div></div>
      <div className="auth-story-art" aria-hidden="true"><div className="auth-art-orbit" /><div className="auth-art-core"><AudioLines size={36} /></div><span className="auth-art-node node-a" /><span className="auth-art-node node-b" /><span className="auth-art-label">FIELD RECORDING <b>· 32 kHz</b></span></div>
      <small className="auth-story-footer">BioSonic · Local research preview</small>
    </section>
    <section className="auth-form-side"><div className="auth-form-wrap"><div className="auth-form-heading"><span className="eyebrow">YOUR WORKSPACE</span><h2>{title}</h2><p>{subtitle}</p></div>{children}</div></section>
  </main>
}

function AuthError({ children }: { children: string }) {
  return children ? <div className="auth-error" role="alert">{children}</div> : null
}

function AuthConnectionError({ message, onRetry, busy }: { message: string; onRetry: () => Promise<void>; busy: boolean }) {
  return message ? <div className="auth-error auth-connection-error" role="alert"><span>{message}</span><button type="button" disabled={busy} onClick={() => void onRetry()}>Retry connection</button></div> : null
}

function passwordMessage(error: unknown) {
  return error instanceof ApiError ? error.message : 'The request could not be completed. Try again.'
}

export function LoginPage() {
  const { user, loading, csrfToken, authError, refresh, login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  if (!loading && user) return <Navigate to="/" replace />

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError('')
    try { await login(email, password); navigate('/', { replace: true }) } catch (reason) { setError(passwordMessage(reason)) } finally { setBusy(false) }
  }

  return <AuthFrame title="Welcome back" subtitle="Sign in to continue to your BioSonic workspace.">
    <form className="auth-form" onSubmit={(event) => void submit(event)}>
      <AuthConnectionError message={authError} onRetry={refresh} busy={loading} />
      <label className="auth-field"><span>Email address</span><span className="auth-input-wrap"><Mail size={16} /><input type="email" autoComplete="email" maxLength={254} required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.org" /></span></label>
      <label className="auth-field"><span>Password</span><span className="auth-input-wrap"><LockKeyhole size={16} /><input type={showPassword ? 'text' : 'password'} autoComplete="current-password" maxLength={128} required value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" /><button type="button" className="password-toggle" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((value) => !value)}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button></span></label>
      <AuthError>{error}</AuthError>
      <button className="auth-submit" type="submit" disabled={busy || loading || !csrfToken}>{busy ? 'Signing in…' : 'Sign in'} <span>→</span></button>
    </form>
    <p className="auth-switch">New to BioSonic? <Link to="/register">Create an account</Link></p>
    <p className="auth-privacy">Audio files are processed for inference and are not retained. Prediction summaries are tied to your account.</p>
  </AuthFrame>
}

export function RegisterPage() {
  const { user, loading, csrfToken, authError, refresh, register } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const passwordHasLength = [...password].length >= 12
  const passwordHasNonSpace = Boolean(password.trim())
  if (!loading && user) return <Navigate to="/" replace />

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError('')
    if (!passwordHasLength || !passwordHasNonSpace) { setError('Use a password with at least 12 characters that is not all whitespace.'); return }
    if (password !== confirm) { setError('The passwords do not match.'); return }
    setBusy(true)
    try { await register(email, password); navigate('/', { replace: true }) } catch (reason) { setError(passwordMessage(reason)) } finally { setBusy(false) }
  }

  return <AuthFrame title="Create your account" subtitle="Your local research workspace starts here.">
    <form className="auth-form" onSubmit={(event) => void submit(event)}>
      <AuthConnectionError message={authError} onRetry={refresh} busy={loading} />
      <label className="auth-field"><span>Email address</span><span className="auth-input-wrap"><Mail size={16} /><input type="email" autoComplete="email" maxLength={254} required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.org" /></span></label>
      <label className="auth-field"><span>Password <small>12 characters minimum</small></span><span className="auth-input-wrap"><LockKeyhole size={16} /><input type={showPassword ? 'text' : 'password'} autoComplete="new-password" minLength={12} maxLength={128} required value={password} onChange={(event) => { setPassword(event.target.value); setError('') }} placeholder="Choose a strong passphrase" /><button type="button" className="password-toggle" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((value) => !value)}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button></span></label>
      {password && <ul className="password-checklist" aria-live="polite"><li className={passwordHasLength ? 'check-valid' : ''}><span aria-hidden="true">{passwordHasLength ? '✓' : '·'}</span> At least 12 characters</li><li className={passwordHasNonSpace ? 'check-valid' : ''}><span aria-hidden="true">{passwordHasNonSpace ? '✓' : '·'}</span> Not entirely whitespace</li></ul>}
      <label className="auth-field"><span>Confirm password</span><span className="auth-input-wrap"><LockKeyhole size={16} /><input type="password" autoComplete="new-password" minLength={12} maxLength={128} required value={confirm} onChange={(event) => { setConfirm(event.target.value); setError('') }} placeholder="Enter it once more" /></span></label>
      <AuthError>{error}</AuthError>
      <button className="auth-submit" type="submit" disabled={busy || loading || !csrfToken}>{busy ? 'Creating account…' : 'Create account'} <span>→</span></button>
    </form>
    <p className="auth-switch">Already registered? <Link to="/login">Sign in</Link></p>
    <p className="auth-privacy">Existing browser-only analysis history is not imported or linked to a new account; it remains unchanged in this browser.</p>
  </AuthFrame>
}

export function AccountPage() {
  const { user, csrfToken, updateEmail, changePassword } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState(user?.email ?? '')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [emailMessage, setEmailMessage] = useState('')
  const [passwordMessageText, setPasswordMessageText] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function saveEmail(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(''); setEmailMessage('')
    try { await updateEmail(email); setEmailMessage('Email address updated.') } catch (reason) { setError(passwordMessage(reason)) } finally { setBusy(false) }
  }

  async function savePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(''); setPasswordMessageText('')
    try {
      await changePassword(currentPassword, newPassword)
      setPasswordMessageText('Password changed. You have been signed out of all sessions.')
      navigate('/login', { replace: true })
    } catch (reason) { setError(passwordMessage(reason)) } finally { setBusy(false) }
  }

  return <div className="page-stack account-page"><div><span className="eyebrow">ACCOUNT SETTINGS</span><h1>Account</h1><p className="page-lede">Manage the email and password for your local BioSonic account.</p></div>
    <div className="account-grid"><section className="panel account-card"><span className="eyebrow">PROFILE</span><h2>Sign-in email</h2><p>This address identifies your account in this local application.</p><form className="account-form" onSubmit={(event) => void saveEmail(event)}><label className="auth-field"><span>Email address</span><input className="plain-input" type="email" autoComplete="email" maxLength={254} required value={email} onChange={(event) => setEmail(event.target.value)} /></label><button className="button-primary" type="submit" disabled={busy || !csrfToken}>Save email</button>{emailMessage && <p className="form-success" role="status">{emailMessage}</p>}</form></section>
    <section className="panel account-card"><span className="eyebrow">SECURITY</span><h2>Change password</h2><p>Changing your password signs out all active sessions, including this one.</p><form className="account-form" onSubmit={(event) => void savePassword(event)}><label className="auth-field"><span>Current password</span><input className="plain-input" type="password" autoComplete="current-password" maxLength={128} required value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} /></label><label className="auth-field"><span>New password <small>12 characters minimum</small></span><input className="plain-input" type="password" autoComplete="new-password" minLength={12} maxLength={128} required value={newPassword} onChange={(event) => setNewPassword(event.target.value)} /></label><AuthError>{error}</AuthError><button className="button-primary" type="submit" disabled={busy || !csrfToken}>{busy ? 'Updating…' : 'Update password'}</button>{passwordMessageText && <p className="form-success" role="status">{passwordMessageText}</p>}</form></section></div>
    <div className="notice-row"><div className="notice-icon"><ShieldCheck size={17} /></div><p><strong>Local account storage.</strong> Account credentials and prediction summaries are stored in the local SQLite database configured for this API. Uploaded audio is not added to history.</p></div>
  </div>
}
