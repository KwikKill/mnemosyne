"use client"

import { useEffect, useRef } from "react"
import { computeEffectiveLR, type LrSchedule } from "@/lib/lr-schedule"

interface LrSchedulePreviewProps {
  schedule: LrSchedule
  baseLR: number
}

const PREVIEW_STEPS = 3000
const SAMPLES = 80

function draw(canvas: HTMLCanvasElement, schedule: LrSchedule, baseLR: number) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
  }
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const width = rect.width
  const height = rect.height
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)

  const range = schedule.type === "cosine" ? Math.max(schedule.totalSteps * 1.15, 200) : PREVIEW_STEPS
  const values = Array.from({ length: SAMPLES }, (_, i) => computeEffectiveLR(schedule, baseLR, (i / (SAMPLES - 1)) * range))
  const max = Math.max(...values, 1e-6)

  const style = getComputedStyle(canvas)
  const accent = style.getPropertyValue("--accent") || "oklch(0.56 0.2 25)"

  ctx.beginPath()
  values.forEach((v, i) => {
    const x = (i / (SAMPLES - 1)) * width
    const y = height - (v / max) * (height - 6) - 3
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.strokeStyle = accent
  ctx.lineWidth = 1.4
  ctx.stroke()
}

// Static preview, not tied to a live run, exists purely so the setup
// screen can show what the schedule will actually do before initiating,
// using the exact same computeEffectiveLR the worker trains with.
export function LrSchedulePreview({ schedule, baseLR }: LrSchedulePreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const render = () => draw(canvas, schedule, baseLR)
    render()
    const observer = new ResizeObserver(render)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [schedule, baseLR])

  return (
    <div className="border border-border h-16 p-1.5">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  )
}
