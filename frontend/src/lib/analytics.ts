import type { StoredPrediction } from '../types/api'

export interface HistoryAnalytics {
  recent: StoredPrediction[]
  predictionsByDay: { date: string; count: number }[]
  classCounts: { label: string; count: number }[]
  durationBands: { band: string; count: number }[]
  averageDurationSeconds: number | null
}

export function buildHistoryAnalytics(items: StoredPrediction[]): HistoryAnalytics {
  const recent = [...items].sort((left, right) => right.created_at - left.created_at)
  const dayCounts = new Map<string, number>()
  const classCounts = new Map<string, number>()
  const durationCounts = new Map<string, number>([
    ['≤ 5 sec', 0], ['5–15 sec', 0], ['15–60 sec', 0], ['> 60 sec', 0],
  ])
  let durationTotal = 0

  for (const item of recent) {
    const date = new Date(item.created_at * 1000).toISOString().slice(0, 10)
    dayCounts.set(date, (dayCounts.get(date) ?? 0) + 1)
    const label = item.response.prediction
    classCounts.set(label, (classCounts.get(label) ?? 0) + 1)
    const duration = item.response.audio.duration_seconds
    durationTotal += duration
    const band = duration <= 5 ? '≤ 5 sec' : duration <= 15 ? '5–15 sec' : duration <= 60 ? '15–60 sec' : '> 60 sec'
    durationCounts.set(band, (durationCounts.get(band) ?? 0) + 1)
  }

  return {
    recent: recent.slice(0, 5),
    predictionsByDay: [...dayCounts].sort(([left], [right]) => left.localeCompare(right)).map(([date, count]) => ({ date, count })),
    classCounts: [...classCounts].sort(([leftLabel, leftCount], [rightLabel, rightCount]) => rightCount - leftCount || leftLabel.localeCompare(rightLabel)).map(([label, count]) => ({ label, count })),
    durationBands: [...durationCounts].map(([band, count]) => ({ band, count })),
    averageDurationSeconds: recent.length ? durationTotal / recent.length : null,
  }
}
