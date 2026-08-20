"use client"

import { type MouseEventHandler, useCallback, useEffect, useRef, useState } from "react"
import { CalibratingOverlay } from "@/components/calibrating-overlay"
import type { TrainingEngine } from "@/lib/engine"
import type { NetworkSnapshot } from "@/lib/nn/network"
import { CLASS_PALETTE } from "@/lib/protocol"
import { cn } from "@/lib/utils"

interface DecisionBoundaryProps {
  engine: TrainingEngine
  probeEnabled?: boolean
  recording?: boolean
  onRecordingReady?: (url: string) => void
  // The replay scrubber's selected historical point, when set, draws (and
  // probes) that instead of the live engine.network. Null/undefined means
  // "live".
  overrideNetwork?: NetworkSnapshot | null
}

interface ProbeMarker {
  id: number
  xPct: number
  yPct: number
  predictedClass: number
  confidence: number
}

const GRID = 56
const MAX_PROBES = 6
const CONTOUR_LEVELS = [0.6, 0.75, 0.9]
// Raw RGB approximation of the theme's --background oklch color, the
// "toward nothing" end of the multi-class confidence shading below.
const BACKGROUND_RGB: [number, number, number] = [22, 12, 9]

function buildGrid(): number[][] {
  const points: number[][] = []
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const x = (gx / (GRID - 1)) * 2 - 1
      const y = (gy / (GRID - 1)) * 2 - 1
      points.push([x, y])
    }
  }
  return points
}

const grid = buildGrid()

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v))
}

function argmax(row: number[]): number {
  let best = 0
  for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k
  return best
}

// Standard marching squares: for each grid cell, compare its 4 corners
// against `level` to pick one of 16 cases, then linearly interpolate the
// crossing point along whichever edges the contour actually passes
// through. Returns segments in fractional grid-cell coordinates (still
// needs scaling to pixels by the caller).
function marchingSquares(field: number[], size: number, level: number): [number, number, number, number][] {
  const segments: [number, number, number, number][] = []
  const at = (gx: number, gy: number) => field[gy * size + gx]
  const interp = (a: number, b: number, va: number, vb: number) => a + ((level - va) / (vb - va)) * (b - a)

  for (let gy = 0; gy < size - 1; gy++) {
    for (let gx = 0; gx < size - 1; gx++) {
      const tl = at(gx, gy)
      const tr = at(gx + 1, gy)
      const br = at(gx + 1, gy + 1)
      const bl = at(gx, gy + 1)
      let idx = 0
      if (tl > level) idx |= 8
      if (tr > level) idx |= 4
      if (br > level) idx |= 2
      if (bl > level) idx |= 1
      if (idx === 0 || idx === 15) continue

      const top: [number, number] = [interp(gx, gx + 1, tl, tr), gy]
      const right: [number, number] = [gx + 1, interp(gy, gy + 1, tr, br)]
      const bottom: [number, number] = [interp(gx, gx + 1, bl, br), gy + 1]
      const left: [number, number] = [gx, interp(gy, gy + 1, tl, bl)]
      const line = (a: [number, number], b: [number, number]) => segments.push([a[0], a[1], b[0], b[1]])

      switch (idx) {
        case 1:
          line(left, bottom)
          break
        case 2:
          line(bottom, right)
          break
        case 3:
          line(left, right)
          break
        case 4:
          line(top, right)
          break
        case 5:
          line(left, top)
          line(bottom, right)
          break
        case 6:
          line(top, bottom)
          break
        case 7:
          line(left, top)
          break
        case 8:
          line(top, left)
          break
        case 9:
          line(top, bottom)
          break
        case 10:
          line(top, right)
          line(left, bottom)
          break
        case 11:
          line(top, right)
          break
        case 12:
          line(right, left)
          break
        case 13:
          line(right, bottom)
          break
        case 14:
          line(bottom, left)
          break
      }
    }
  }
  return segments
}

function draw(canvas: HTMLCanvasElement, offscreen: HTMLCanvasElement, engine: TrainingEngine, overrideNetwork?: NetworkSnapshot | null) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
  }
  const ctx = canvas.getContext("2d")
  const octx = offscreen.getContext("2d")
  if (!ctx || !octx) return

  const network = overrideNetwork ?? engine.network
  const isBinary = engine.dataset.numClasses === 2
  const predictions = network.predict(grid)
  const imageData = octx.createImageData(GRID, GRID)
  const confidence = new Array(grid.length)

  for (let i = 0; i < grid.length; i++) {
    const row = predictions[i]
    let r: number
    let g: number
    let b: number
    if (isBinary) {
      const p = row[0]
      const [c0r, c0g, c0b] = CLASS_PALETTE[0]
      const [c1r, c1g, c1b] = CLASS_PALETTE[1]
      r = c0r + (c1r - c0r) * p
      g = c0g + (c1g - c0g) * p
      b = c0b + (c1b - c0b) * p
      confidence[i] = Math.max(p, 1 - p)
    } else {
      const numClasses = row.length
      const winner = argmax(row)
      const conf = row[winner]
      confidence[i] = conf
      const t = clamp01((conf - 1 / numClasses) / (1 - 1 / numClasses))
      const shade = 0.15 + 0.85 * t
      const cls = CLASS_PALETTE[winner] ?? CLASS_PALETTE[0]
      r = BACKGROUND_RGB[0] + (cls[0] - BACKGROUND_RGB[0]) * shade
      g = BACKGROUND_RGB[1] + (cls[1] - BACKGROUND_RGB[1]) * shade
      b = BACKGROUND_RGB[2] + (cls[2] - BACKGROUND_RGB[2]) * shade
    }
    const idx = i * 4
    imageData.data[idx] = r
    imageData.data[idx + 1] = g
    imageData.data[idx + 2] = b
    imageData.data[idx + 3] = 255
  }
  octx.putImageData(imageData, 0, 0)

  const width = rect.width
  const height = rect.height
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  ctx.globalAlpha = 0.55
  ctx.drawImage(offscreen, 0, 0, GRID, GRID, 0, 0, width, height)
  ctx.globalAlpha = 1

  // Confidence contours: same "distance from certainty" field for binary
  // and multi-class alike (max(p, 1-p) vs. max of the softmax row), so
  // this loop doesn't fork on numClasses the way the fill above does.
  const style = getComputedStyle(canvas)
  const contourColor = style.getPropertyValue("--foreground") || "oklch(0.92 0.03 55)"
  ctx.strokeStyle = contourColor
  ctx.lineWidth = 0.75
  for (const [i, level] of CONTOUR_LEVELS.entries()) {
    ctx.globalAlpha = 0.18 + i * 0.12
    for (const [x1, y1, x2, y2] of marchingSquares(confidence, GRID, level)) {
      ctx.beginPath()
      ctx.moveTo((x1 / (GRID - 1)) * width, (y1 / (GRID - 1)) * height)
      ctx.lineTo((x2 / (GRID - 1)) * width, (y2 / (GRID - 1)) * height)
      ctx.stroke()
    }
  }
  ctx.globalAlpha = 1

  const toPx = (x: number, y: number) => [((x + 1) / 2) * width, ((y + 1) / 2) * height]
  const numClasses = engine.dataset.numClasses
  for (let i = 0; i < engine.dataset.X.length; i++) {
    const [x, y] = engine.dataset.X[i]
    const label = engine.dataset.Y[i]
    const cls = numClasses === 2 ? label[0] : argmax(label)
    const [px, py] = toPx(x, y)
    ctx.beginPath()
    ctx.arc(px, py, 2.4, 0, Math.PI * 2)
    const [r, g, b] = CLASS_PALETTE[cls] ?? CLASS_PALETTE[0]
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
    ctx.strokeStyle = "rgba(0,0,0,0.5)"
    ctx.lineWidth = 0.5
    ctx.fill()
    ctx.stroke()
  }
}

let probeIdCounter = 0

export function DecisionBoundary({
  engine,
  probeEnabled = false,
  recording = false,
  onRecordingReady,
  overrideNetwork,
}: DecisionBoundaryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const offscreenRef = useRef<HTMLCanvasElement | null>(null)
  const [probes, setProbes] = useState<ProbeMarker[]>([])
  const [ready, setReady] = useState(false)
  const [reticlePos, setReticlePos] = useState<{ x: number; y: number } | null>(null)

  // Records this canvas as a downloadable clip of the run's convergence.
  // Starts when `recording` flips true (a run entering "running"), stops
  // via this effect's cleanup when it flips false (halt), which is what
  // fires MediaRecorder's onstop and hands the finished blob URL up.
  // Feature-detected: silently does nothing on a browser without
  // captureStream/MediaRecorder rather than erroring.
  useEffect(() => {
    if (!recording) return
    const canvas = canvasRef.current
    if (!canvas || typeof canvas.captureStream !== "function" || typeof MediaRecorder === "undefined") return

    const stream = canvas.captureStream(10)
    let recorder: MediaRecorder
    try {
      recorder = new MediaRecorder(stream, { mimeType: "video/webm" })
    } catch {
      return
    }
    const chunks: BlobPart[] = []
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }
    recorder.onstop = () => {
      if (chunks.length === 0) return
      const blob = new Blob(chunks, { type: "video/webm" })
      onRecordingReady?.(URL.createObjectURL(blob))
    }
    recorder.start()
    return () => {
      if (recorder.state !== "inactive") recorder.stop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording])

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement("canvas")
      offscreenRef.current.width = GRID
      offscreenRef.current.height = GRID
    }
    draw(canvas, offscreenRef.current, engine, overrideNetwork)
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

  // A fresh halted state is a fresh probe surface, so clear old markers each
  // time probing newly becomes available rather than letting stale probes
  // from a previous run linger over a different trained network.
  useEffect(() => {
    if (probeEnabled) setProbes([])
  }, [probeEnabled])

  const handleClick: MouseEventHandler<HTMLDivElement> = (e) => {
    if (!probeEnabled) return
    const rect = e.currentTarget.getBoundingClientRect()
    const xPct = ((e.clientX - rect.left) / rect.width) * 100
    const yPct = ((e.clientY - rect.top) / rect.height) * 100
    const x = (xPct / 100) * 2 - 1
    const y = (yPct / 100) * 2 - 1
    const [row] = (overrideNetwork ?? engine.network).predict([[x, y]])
    const isBinary = engine.dataset.numClasses === 2
    const predictedClass = isBinary ? (row[0] >= 0.5 ? 1 : 0) : argmax(row)
    const confidence = isBinary ? (predictedClass === 1 ? row[0] : 1 - row[0]) : row[predictedClass]
    setProbes((prev) => [
      ...prev.slice(-(MAX_PROBES - 1)),
      { id: probeIdCounter++, xPct, yPct, predictedClass, confidence },
    ])
  }

  const handleMouseMove: MouseEventHandler<HTMLDivElement> = (e) => {
    if (!probeEnabled) return
    const rect = e.currentTarget.getBoundingClientRect()
    setReticlePos({ x: ((e.clientX - rect.left) / rect.width) * 100, y: ((e.clientY - rect.top) / rect.height) * 100 })
  }

  return (
    <div className="mnemo-panel h-full min-h-48 p-3 flex flex-col gap-2">
      <div className="mnemo-header">
        <span>Classification Surface</span>
        {probeEnabled && <span className="text-accent">PROBE ACTIVE</span>}
      </div>
      <div
        className={cn("relative flex-1", probeEnabled && "cursor-none")}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setReticlePos(null)}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ imageRendering: "auto" }} />
        {!ready && <CalibratingOverlay />}
        {probeEnabled && reticlePos && (
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none"
            style={{ left: `${reticlePos.x}%`, top: `${reticlePos.y}%` }}
          >
            <div className="relative size-6">
              <span className="absolute left-1/2 top-0 -translate-x-1/2 w-px h-2 bg-accent" />
              <span className="absolute left-1/2 bottom-0 -translate-x-1/2 w-px h-2 bg-accent" />
              <span className="absolute top-1/2 left-0 -translate-y-1/2 h-px w-2 bg-accent" />
              <span className="absolute top-1/2 right-0 -translate-y-1/2 h-px w-2 bg-accent" />
              <span className="absolute inset-1.75 rounded-full border border-accent" style={{ boxShadow: "0 0 6px 1px var(--accent)" }} />
            </div>
          </div>
        )}
        {probes.map((probe) => {
          const [r, g, b] = CLASS_PALETTE[probe.predictedClass] ?? CLASS_PALETTE[0]
          const rgb = `rgb(${r}, ${g}, ${b})`
          return (
            <div
              key={probe.id}
              className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none flex flex-col items-center"
              style={{ left: `${probe.xPct}%`, top: `${probe.yPct}%` }}
            >
              <span
                className="block size-2 rounded-full border"
                style={{ backgroundColor: rgb, borderColor: rgb, boxShadow: `0 0 6px 1px ${rgb}` }}
              />
              <span
                className="mt-1 whitespace-nowrap text-[0.6rem] px-1 py-0.5 bg-background/90 border"
                style={{ color: rgb, borderColor: rgb }}
              >
                CLASS {probe.predictedClass} · {(probe.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
