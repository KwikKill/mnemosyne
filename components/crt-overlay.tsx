"use client"

// Decorative station-monitor dressing: scanline texture, a slow sweeping
// scan bar, and a continuous rounded bezel, closer to the reference HUD
// captures' physical device screen (one unbroken rounded frame) than a
// CRT's sharp-cornered glass, plus small corner ticks for detail.
//
// Deliberately `fixed` to the viewport (not `absolute` on the page), the
// bezel stays put like a physical monitor frame while the page scrolls
// underneath it. That means page content needs its own bottom clearance so
// the last bit of it doesn't scroll up underneath the fixed frame's bottom
// border, see the padding-bottom on <main> in app/page.tsx.
import { useEffect, useRef, useState } from "react"
import type { TrainingEngine } from "@/lib/engine"
import { cn } from "@/lib/utils"

interface CrtOverlayProps {
  engine?: TrainingEngine
}

const CORNERS = [
  "top-0 left-0 border-t-[2px] border-l-[2px] rounded-tl-2xl",
  "top-0 right-0 border-t-[2px] border-r-[2px] rounded-tr-2xl",
  "bottom-0 left-0 border-b-[2px] border-l-[2px] rounded-bl-2xl",
  "bottom-0 right-0 border-b-[2px] border-r-[2px] rounded-br-2xl",
]

// A jump this much larger than the previous tick's loss (and not just
// noise near zero) reads as real training instability, not float jitter.
const SPIKE_RELATIVE = 1.25
const SPIKE_ABSOLUTE = 0.01
const GLITCH_MS = 420

export function CrtOverlay({ engine }: CrtOverlayProps) {
  const [glitching, setGlitching] = useState(false)
  const lastLossRef = useRef<number | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Ties the decorative frame to real training instability instead of
  // looping ambiently: a loss spike briefly intensifies the glitch, then
  // settles back, rather than playing on a timer regardless of what the
  // network is actually doing.
  useEffect(() => {
    if (!engine) return
    return engine.subscribe((tick) => {
      const last = lastLossRef.current
      if (last !== null && tick.loss > last * SPIKE_RELATIVE && tick.loss - last > SPIKE_ABSOLUTE) {
        setGlitching(true)
        if (timeoutRef.current) clearTimeout(timeoutRef.current)
        timeoutRef.current = setTimeout(() => setGlitching(false), GLITCH_MS)
      }
      lastLossRef.current = tick.loss
    })
  }, [engine])

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [])

  return (
    <div
      aria-hidden="true"
      className={cn(
        "pointer-events-none fixed inset-2 sm:inset-4 z-50",
        glitching && "animate-[mnemo-glitch_0.4s_ease-out]",
      )}
    >
      <div className="absolute inset-0 overflow-hidden rounded-2xl sm:rounded-3xl">
        <div
          className="absolute inset-0 mix-blend-overlay transition-opacity duration-200"
          style={{
            opacity: glitching ? 0.6 : 0.35,
            backgroundImage:
              "repeating-linear-gradient(0deg, rgba(0,0,0,0.5) 0px, rgba(0,0,0,0.5) 1px, transparent 1px, transparent 3px)",
          }}
        />
        <div
          className="absolute inset-x-0 top-0 h-px animate-[mnemo-scan_9s_linear_infinite]"
          style={{ backgroundColor: "var(--accent)", boxShadow: "0 0 16px 2px var(--accent)" }}
        />
      </div>

      <div
        className="absolute inset-0 rounded-2xl sm:rounded-3xl transition-shadow duration-200"
        style={{
          border: `2px solid color-mix(in oklch, var(--accent) ${glitching ? 90 : 60}%, transparent)`,
          boxShadow: `0 0 0 1px color-mix(in oklch, var(--accent) 12%, transparent) inset, 0 0 ${glitching ? 44 : 28}px color-mix(in oklch, var(--accent) ${glitching ? 60 : 40}%, transparent)`,
        }}
      />
      <div
        className="absolute inset-2 rounded-xl sm:rounded-2xl"
        style={{ border: "1px solid color-mix(in oklch, var(--accent) 25%, transparent)" }}
      />

      {CORNERS.map((pos) => (
        <span
          key={pos}
          className={`absolute size-8 sm:size-10 ${pos}`}
          style={{ borderColor: "var(--accent)", filter: "drop-shadow(0 0 4px var(--accent))" }}
        />
      ))}
    </div>
  )
}
