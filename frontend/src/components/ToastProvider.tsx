import { useState, type ReactNode } from 'react'
import { X } from 'lucide-react'
import { ToastContext, type ToastKind } from '../lib/toastContext'

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<{ message: string; kind: ToastKind } | null>(null)
  const showToast = (message: string, kind: ToastKind = 'info') => {
    setToast({ message, kind })
    window.setTimeout(() => setToast(null), 4800)
  }

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {toast && (
        <div className={`toast toast-${toast.kind}`} role={toast.kind === 'error' ? 'alert' : 'status'}>
          <span>{toast.message}</span>
          <button type="button" aria-label="Dismiss notification" onClick={() => setToast(null)}><X size={16} /></button>
        </div>
      )}
    </ToastContext.Provider>
  )
}
