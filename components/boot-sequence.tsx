"use client"

import { useEffect, useRef, useState } from "react"

interface BootSequenceProps {
  onComplete: () => void
}

const LINES = [
  "MNEMOSYNE DIAGNOSTIC TERMINAL v0.6.1",
  "INITIALIZING CORE MODULES...",
  "  [OK] MATRIX ENGINE",
  "  [OK] BACKPROPAGATION UNIT",
  "  [OK] SYNAPTIC RENDERER",
  "  [OK] TRAINING WORKER THREAD",
  "CALIBRATING SENSORS...",
  "STATION-7 OBSERVATION DECK READY",
]

const LINE_DELAY_MS = 220
const HOLD_MS = 500

export function BootSequence({ onComplete }: BootSequenceProps) {
  const [visibleCount, setVisibleCount] = useState(0)
  const completedRef = useRef(false)

  const finish = () => {
    if (completedRef.current) return
    completedRef.current = true
    onComplete()
  }

  useEffect(() => {
    if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      finish()
      return
    }
    if (visibleCount >= LINES.length) {
      const holdTimer = setTimeout(finish, HOLD_MS)
      return () => clearTimeout(holdTimer)
    }
    const timer = setTimeout(() => setVisibleCount((c) => c + 1), LINE_DELAY_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleCount])

  return (
    <div
      className="fixed inset-0 z-50 bg-background flex items-center justify-center cursor-pointer p-6"
      onClick={finish}
    >
      <div className="max-w-md w-full font-mono text-xs sm:text-sm text-accent leading-relaxed">
        {LINES.slice(0, visibleCount).map((line, i) => (
          <p key={i} className={line.startsWith(" ") ? "text-muted-foreground" : "text-accent"}>
            {line}
          </p>
        ))}
        <span className="inline-block w-2 h-3.5 bg-accent align-middle animate-[mnemo-caret_1s_steps(1)_infinite]" />
        <p className="mnemo-label mt-6 normal-case tracking-normal text-[0.65rem]">click to skip</p>
      </div>
    </div>
  )
}
