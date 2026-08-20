"use client"

import { type PointerEventHandler, useEffect, useRef, useState } from "react"
import { Toggle } from "@/components/field-controls"
import { type CustomPoint, MIN_POINTS_PER_CLASS } from "@/lib/nn/datasets"
import { CLASS_PALETTE } from "@/lib/protocol"
import { cn } from "@/lib/utils"

interface DatasetPainterProps {
  points: CustomPoint[]
  onPointsChange: (points: CustomPoint[]) => void
  numClasses: number
  onNumClassesChange: (n: number) => void
}

// Minimum distance (in the [-1,1] data space) between two points placed by
// the same drag stroke, keeps a slow drag from spamming hundreds of
// near-duplicate points.
const MIN_SPACING = 0.05

function rgb([r, g, b]: [number, number, number]) {
  return `rgb(${r}, ${g}, ${b})`
}

function rgba([r, g, b]: [number, number, number], alpha: number) {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function draw(canvas: HTMLCanvasElement, points: CustomPoint[]) {
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

  for (const p of points) {
    const px = ((p.x + 1) / 2) * width
    const py = ((p.y + 1) / 2) * height
    ctx.beginPath()
    ctx.arc(px, py, 3, 0, Math.PI * 2)
    ctx.fillStyle = rgb(CLASS_PALETTE[p.cls] ?? CLASS_PALETTE[0])
    ctx.strokeStyle = "rgba(0,0,0,0.5)"
    ctx.lineWidth = 0.6
    ctx.fill()
    ctx.stroke()
  }
}

export function DatasetPainter({ points, onPointsChange, numClasses, onNumClassesChange }: DatasetPainterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [activeClass, setActiveClass] = useState(0)
  const paintingRef = useRef(false)
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const render = () => draw(canvas, points)
    render()
    const observer = new ResizeObserver(render)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [points])

  const addPointAt = (clientX: number, clientY: number, rect: DOMRect) => {
    const x = ((clientX - rect.left) / rect.width) * 2 - 1
    const y = ((clientY - rect.top) / rect.height) * 2 - 1
    const last = lastPointRef.current
    if (last) {
      const dx = x - last.x
      const dy = y - last.y
      if (Math.sqrt(dx * dx + dy * dy) < MIN_SPACING) return
    }
    lastPointRef.current = { x, y }
    onPointsChange([...points, { x, y, cls: activeClass }])
  }

  const handlePointerDown: PointerEventHandler<HTMLDivElement> = (e) => {
    paintingRef.current = true
    lastPointRef.current = null
    addPointAt(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect())
  }
  const handlePointerMove: PointerEventHandler<HTMLDivElement> = (e) => {
    if (!paintingRef.current) return
    addPointAt(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect())
  }
  const stopPainting = () => {
    paintingRef.current = false
    lastPointRef.current = null
  }

  const counts = Array.from({ length: numClasses }, (_, k) => points.filter((p) => p.cls === k).length)
  const allValid = counts.every((c) => c >= MIN_POINTS_PER_CLASS)

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <div className="mnemo-label">Class Count</div>
        <Toggle
          options={[2, 3, 4].map((n) => ({ id: String(n), label: String(n) }))}
          value={String(numClasses)}
          onChange={(v) => {
            const n = Number(v)
            onNumClassesChange(n)
            onPointsChange(points.filter((p) => p.cls < n))
            if (activeClass >= n) setActiveClass(0)
          }}
        />
      </div>

      <div className="flex gap-1.5">
        {Array.from({ length: numClasses }, (_, k) => (
          <button
            key={k}
            onClick={() => setActiveClass(k)}
            className="flex-1 border px-2 py-1.5 text-[0.65rem] tracking-wider cursor-pointer transition-opacity"
            style={{
              borderColor: rgb(CLASS_PALETTE[k]),
              color: rgb(CLASS_PALETTE[k]),
              backgroundColor: activeClass === k ? rgba(CLASS_PALETTE[k], 0.15) : "transparent",
              opacity: activeClass === k ? 1 : 0.55,
            }}
          >
            CLASS {k} · {counts[k]}
          </button>
        ))}
      </div>

      <div
        className={cn("relative border border-border aspect-square touch-none cursor-crosshair")}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={stopPainting}
        onPointerLeave={stopPainting}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      </div>

      <div className="flex items-center justify-between gap-2">
        <p className="mnemo-label normal-case tracking-normal text-[0.65rem]">
          {allValid ? `${points.length} points painted` : `minimum ${MIN_POINTS_PER_CLASS} points per class required`}
        </p>
        <div className="flex gap-1.5">
          <button
            onClick={() => onPointsChange(points.slice(0, -1))}
            disabled={points.length === 0}
            className="border border-border text-muted-foreground px-2 py-1 text-[0.65rem] tracking-wider cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:border-accent/50 hover:text-foreground"
          >
            UNDO
          </button>
          <button
            onClick={() => onPointsChange([])}
            disabled={points.length === 0}
            className="border border-alert/60 text-alert px-2 py-1 text-[0.65rem] tracking-wider cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:bg-alert/15"
          >
            CLEAR
          </button>
        </div>
      </div>
    </div>
  )
}
