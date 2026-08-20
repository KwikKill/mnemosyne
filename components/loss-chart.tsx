"use client"

import { useEffect, useRef, useState } from "react"
import { CalibratingOverlay } from "@/components/calibrating-overlay"
import type { EngineTick, TrainingEngine } from "@/lib/engine"

interface LossChartProps {
  engine: TrainingEngine
}

const HISTORY_LEN = 240

interface LastTick {
  step: number
  loss: number
  valLoss?: number
}

function drawLine(ctx: CanvasRenderingContext2D, history: number[], max: number, width: number, height: number, color: string, fill: boolean) {
  if (history.length < 2) return
  const stepX = width / (HISTORY_LEN - 1)
  const offset = HISTORY_LEN - history.length

  ctx.beginPath()
  history.forEach((loss, i) => {
    const x = (offset + i) * stepX
    const y = height - (loss / max) * (height - 8) - 4
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.strokeStyle = color
  ctx.lineWidth = 1.4
  ctx.stroke()

  if (fill) {
    ctx.lineTo((offset + history.length - 1) * stepX, height)
    ctx.lineTo(offset * stepX, height)
    ctx.closePath()
    ctx.globalAlpha = 0.12
    ctx.fillStyle = color
    ctx.fill()
    ctx.globalAlpha = 1
  }
}

function draw(canvas: HTMLCanvasElement, trainHistory: number[], valHistory: number[], tick: LastTick | null) {
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

  const style = getComputedStyle(canvas)
  const accent = style.getPropertyValue("--accent") || "oklch(0.56 0.2 25)"
  const foreground = style.getPropertyValue("--foreground") || "oklch(0.92 0.03 55)"
  const muted = style.getPropertyValue("--muted-foreground") || "oklch(0.58 0.07 30)"

  const max = Math.max(...trainHistory, ...valHistory, 1e-3)
  drawLine(ctx, trainHistory, max, width, height, accent, true)
  drawLine(ctx, valHistory, max, width, height, foreground, false)

  ctx.font = "11px var(--font-mono, monospace)"
  ctx.fillStyle = accent
  ctx.textBaseline = "top"
  ctx.fillText(tick ? `L ${tick.loss.toFixed(4)}` : "L N/A", 4, 4)
  if (tick?.valLoss !== undefined) {
    ctx.fillStyle = foreground
    ctx.fillText(`VAL ${tick.valLoss.toFixed(4)}`, 4, 17)
  }
  ctx.fillStyle = muted
  ctx.textAlign = "right"
  ctx.fillText(tick ? `STEP ${tick.step}` : "STEP 0", width - 4, 4)
  ctx.textAlign = "left"
}

export function LossChart({ engine }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trainHistoryRef = useRef<number[]>([])
  const valHistoryRef = useRef<number[]>([])
  const lastTickRef = useRef<LastTick | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const render = () => draw(canvas, trainHistoryRef.current, valHistoryRef.current, lastTickRef.current)
    const onTick = (tick: EngineTick) => {
      setReady(true)
      if (tick.events.some((e) => e.type === "reset")) {
        trainHistoryRef.current = []
        valHistoryRef.current = []
      }
      trainHistoryRef.current.push(tick.loss)
      if (trainHistoryRef.current.length > HISTORY_LEN) trainHistoryRef.current.shift()
      if (tick.valLoss !== undefined) {
        valHistoryRef.current.push(tick.valLoss)
        if (valHistoryRef.current.length > HISTORY_LEN) valHistoryRef.current.shift()
      }
      lastTickRef.current = { step: tick.step, loss: tick.loss, valLoss: tick.valLoss }
      render()
    }
    render()
    const unsubscribe = engine.subscribe(onTick)
    const observer = new ResizeObserver(render)
    observer.observe(canvas)
    return () => {
      unsubscribe()
      observer.disconnect()
    }
  }, [engine])

  return (
    <div className="mnemo-panel h-full min-h-24 p-3 flex flex-col gap-2">
      <div className="mnemo-header">Loss Telemetry</div>
      <div className="relative flex-1">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        {!ready && <CalibratingOverlay />}
      </div>
    </div>
  )
}
