import { createContext, useContext } from 'react'

export type ToastKind = 'success' | 'error' | 'info'
export type ToastContextValue = { showToast: (message: string, kind?: ToastKind) => void }

export const ToastContext = createContext<ToastContextValue>({ showToast: () => undefined })
export const useToast = () => useContext(ToastContext)
