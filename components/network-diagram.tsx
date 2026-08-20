"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { CalibratingOverlay } from "@/components/calibrating-overlay"
import type { TrainingEngine } from "@/lib/engine"
import type { Matrix } from "@/lib/nn/matrix"
import type { NetworkSnapshot } from "@/lib/nn/network"

interface NetworkDiagramProps {
  engine: TrainingEngine
  // The replay scrubber's selected historical point, when set, draws that
  // instead of the live engine.network. Null/undefined means "live".
  overrideNetwork?: NetworkSnapshot | null
}

function meanAbsColumns(m: Matrix): number[] {
  const cols = m[0].length
  const out = new Array(cols).fill(0)
  for (const row of m) for (let j = 0; j < cols; j++) out[j] += Math.abs(row[j])
  return out.map((v) => v / m.length)
}

function draw(canvas: HTMLCanvasElement, engine: TrainingEngine, overrideNetwork?: NetworkSnapshot | null) {
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

  const network = overrideNetwork ?? engine.network
  const sizes = network.sizes
  const padX = 36
  const padY = 24
  const colGap = sizes.length > 1 ? (width - padX * 2) / (sizes.length - 1) : 0
  const positions: { x: number; y: number }[][] = sizes.map((size, li) => {
    const x = padX + colGap * li
    const gap = size > 1 ? (height - padY * 2) / (size - 1) : 0
    return Array.from({ length: size }, (_, ni) => ({
      x,
      y: size === 1 ? height / 2 : padY + gap * ni,
    }))
  })

  const activations = network.lastActivations
  const nodeIntensity: number[][] = sizes.map((size, li) => {
    if (!activations || !activations[li]) return new Array(size).fill(0.15)
    const means = meanAbsColumns(activations[li])
    const max = Math.max(...means, 1e-6)
    return means.map((v) => 0.15 + 0.85 * (v / max))
  })

  const style = getComputedStyle(canvas)
  const accent = style.getPropertyValue("--accent") || "oklch(0.56 0.2 25)"
  const foreground = style.getPropertyValue("--foreground") || "oklch(0.92 0.03 55)"
  const nodeFill = style.getPropertyValue("--background") || "oklch(0.1 0.018 25)"

  // edges
  network.layers.forEach((layer, li) => {
    const from = positions[li]
    const to = positions[li + 1]
    const flat = layer.weights.flat()
    const maxAbs = Math.max(...flat.map(Math.abs), 1e-6)
    for (let i = 0; i < layer.weights.length; i++) {
      for (let j = 0; j < layer.weights[i].length; j++) {
        const w = layer.weights[i][j]
        const strength = Math.abs(w) / maxAbs
        if (strength < 0.05) continue
        ctx.beginPath()
        ctx.moveTo(from[i].x, from[i].y)
        ctx.lineTo(to[j].x, to[j].y)
        ctx.strokeStyle = w >= 0 ? accent : foreground
        ctx.globalAlpha = 0.12 + 0.55 * strength
        ctx.lineWidth = 0.4 + strength * 1.8
        ctx.stroke()
      }
    }
  })
  ctx.globalAlpha = 1

  // nodes
  positions.forEach((layer, li) => {
    layer.forEach((pos, ni) => {
      const intensity = nodeIntensity[li][ni]
      const r = 4.5
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2)
      ctx.fillStyle = accent
      ctx.globalAlpha = intensity * 0.25
      ctx.fill()

      ctx.beginPath()
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2)
      ctx.globalAlpha = 1
      ctx.fillStyle = nodeFill
      ctx.fill()
      ctx.lineWidth = 1
      ctx.strokeStyle = accent
      ctx.globalAlpha = 0.35 + intensity * 0.65
      ctx.stroke()
      ctx.globalAlpha = 1
    })
  })
}

export function NetworkDiagram({ engine, overrideNetwork }: NetworkDiagramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [ready, setReady] = useState(false)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    draw(canvas, engine, overrideNetwork)
  }, [engine, overrideNetwork])

  // Subscribing to the engine is only ever about live ticks, it doesn't
  // need to restart just because the scrubber moved, so it stays keyed on
  // `engine` alone rather than including `render` (which changes identity
  // whenever `overrideNetwork` does).
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine])

  // Redraws immediately when the replay scrubber's selection changes,
  // independent of live ticks (the run is halted while this is possible).
  useEffect(() => {
    render()
  }, [render])

  return (
    <div className="mnemo-panel h-full min-h-64 p-3 flex flex-col gap-2">
      <div className="mnemo-header">
        <span>Synaptic Array</span>
        <span>{(overrideNetwork ?? engine.network).sizes.join(" · ")}</span>
      </div>
      <div className="relative flex-1">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        {!ready && <CalibratingOverlay />}
      </div>
    </div>
  )
}
