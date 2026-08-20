"use client"

import { useEffect, useRef, useState } from "react"
import { CalibratingOverlay } from "@/components/calibrating-overlay"
import type { TrainingEngine } from "@/lib/engine"

interface WeightHistogramProps {
  engine: TrainingEngine
}

const BIN_COUNT = 22

function draw(canvas: HTMLCanvasElement, engine: TrainingEngine) {
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

  const weights: number[] = []
  for (const layer of engine.network.layers) {
    for (const row of layer.weights) for (const w of row) weights.push(w)
  }

  const style = getComputedStyle(canvas)
  const accent = style.getPropertyValue("--accent") || "oklch(0.56 0.2 25)"
  const muted = style.getPropertyValue("--muted-foreground") || "oklch(0.58 0.07 30)"

  if (weights.length > 0) {
    const min = Math.min(...weights)
    const max = Math.max(...weights)
    const span = max - min || 1
    const bins = new Array(BIN_COUNT).fill(0)
    for (const w of weights) {
      const bin = Math.min(BIN_COUNT - 1, Math.floor(((w - min) / span) * BIN_COUNT))
      bins[bin]++
    }
    const maxCount = Math.max(...bins, 1)
    const chartTop = 18
    const chartHeight = height - chartTop - 4
    const barWidth = width / BIN_COUNT

    bins.forEach((count, i) => {
      const barHeight = (count / maxCount) * chartHeight
      ctx.fillStyle = accent
      ctx.globalAlpha = 0.25 + 0.65 * (count / maxCount)
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, barWidth - 1, barHeight)
    })
    ctx.globalAlpha = 1

    ctx.font = "11px var(--font-mono, monospace)"
    ctx.fillStyle = accent
    ctx.textBaseline = "top"
    ctx.fillText(`W N=${weights.length}`, 4, 4)
    ctx.fillStyle = muted
    ctx.textAlign = "right"
    ctx.fillText(`${min.toFixed(2)} .. ${max.toFixed(2)}`, width - 4, 4)
    ctx.textAlign = "left"
  }
}

export function WeightHistogram({ engine }: WeightHistogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const render = () => draw(canvas, engine)
    render()
    const unsubscribe = engine.subscribe(() => {
      setReady(true)
      render()
    })
    const observer = new ResizeObserver(render)
    observer.observe(canvas)
    return () => {
      unsubscribe()
      observer.disconnect()
    }
  }, [engine])

  return (
    <div className="mnemo-panel h-full min-h-24 p-3 flex flex-col gap-2">
      <div className="mnemo-header">Weight Distribution</div>
      <div className="relative flex-1">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        {!ready && <CalibratingOverlay />}
      </div>
    </div>
  )
}
