"use client"

import { cn } from "@/lib/utils"

export function Toggle<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
}) {
  return (
    <div className="flex gap-1">
      {options.map((opt) => (
        <button
          key={opt.id}
          onClick={() => onChange(opt.id)}
          className={cn(
            "flex-1 border px-2 py-1 text-[0.65rem] tracking-wider transition-colors cursor-pointer",
            value === opt.id
              ? "border-accent bg-accent/15 text-accent"
              : "border-border text-muted-foreground hover:border-accent/50 hover:text-foreground",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

export function Slider({
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  display: string
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="mnemo-label flex justify-between">
        <span>{label}</span>
        <span className="text-accent">{display}</span>
      </div>
      <input
        type="range"
        className="mnemo-range w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
