"use client"

import type { HistorySnapshot } from "@/lib/engine"

interface ReplayScrubberProps {
  history: HistorySnapshot[] | null
  index: number | null
  onChange: (index: number | null) => void
}

export function ReplayScrubber({ history, index, onChange }: ReplayScrubberProps) {
  if (!history || history.length < 2) return null

  const lastIndex = history.length - 1
  const current = index ?? lastIndex
  const point = history[current]
  const isLive = index === null

  return (
    <div className="flex flex-col gap-1.5 border-t border-border pt-3">
      <div className="mnemo-label flex justify-between">
        <span>Training Replay</span>
        <span className="text-accent">{isLive ? "LIVE" : `T+${String(point.step).padStart(6, "0")}`}</span>
      </div>
      <input
        type="range"
        className="mnemo-range w-full"
        min={0}
        max={lastIndex}
        step={1}
        value={current}
        onChange={(e) => {
          const v = Number(e.target.value)
          onChange(v >= lastIndex ? null : v)
        }}
      />
      <div className="flex items-center justify-between">
        <span className="mnemo-label normal-case tracking-normal text-[0.6rem]">loss {point.loss.toFixed(4)}</span>
        {!isLive && (
          <button
            onClick={() => onChange(null)}
            className="border border-border text-muted-foreground px-2 py-0.5 text-[0.6rem] tracking-wider cursor-pointer hover:border-accent/50 hover:text-foreground"
          >
            RETURN TO LIVE
          </button>
        )}
      </div>
    </div>
  )
}
