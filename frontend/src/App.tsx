import { lazy, Suspense } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router'
import { AuthProvider } from './components/AuthProvider'
import { PublicOnly, RequireAuth } from './components/AuthGuards'
import { AppShell } from './components/AppShell'
import { AccountPage, LoginPage, RegisterPage } from './pages/AuthPages'
import { HistoryPage } from './pages/HistoryPage'

const AboutPage = lazy(() => import('./pages/AboutPage').then((module) => ({ default: module.AboutPage })))
const AnalysisPage = lazy(() => import('./pages/AnalysisPage').then((module) => ({ default: module.AnalysisPage })))
const BatchAnalysisPage = lazy(() => import('./pages/BatchAnalysisPage').then((module) => ({ default: module.BatchAnalysisPage })))
const DashboardPage = lazy(() => import('./pages/DashboardPage').then((module) => ({ default: module.DashboardPage })))
const ModelsPage = lazy(() => import('./pages/ModelsPage').then((module) => ({ default: module.ModelsPage })))
const ResultsPage = lazy(() => import('./pages/ResultsPage').then((module) => ({ default: module.ResultsPage })))

function PageLoading() {
  return <div className="page-loading" role="status"><span className="spinner" /> Loading BioSonic view…</div>
}

function NotFound() {
  return <div className="not-found"><span className="eyebrow">NOT FOUND</span><h1>This trail ends here.</h1><Link className="button-primary" to="/">Return to overview</Link></div>
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="login" element={<PublicOnly><LoginPage /></PublicOnly>} />
          <Route path="register" element={<PublicOnly><RegisterPage /></PublicOnly>} />
          <Route element={<RequireAuth />}>
            <Route element={<AppShell />}>
              <Route index element={<Suspense fallback={<PageLoading />}><DashboardPage /></Suspense>} />
              <Route path="analysis" element={<Suspense fallback={<PageLoading />}><AnalysisPage /></Suspense>} />
              <Route path="batch" element={<Suspense fallback={<PageLoading />}><BatchAnalysisPage /></Suspense>} />
              <Route path="results/:id" element={<Suspense fallback={<PageLoading />}><ResultsPage /></Suspense>} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="account" element={<AccountPage />} />
              <Route path="models" element={<Suspense fallback={<PageLoading />}><ModelsPage /></Suspense>} />
              <Route path="about" element={<Suspense fallback={<PageLoading />}><AboutPage /></Suspense>} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
