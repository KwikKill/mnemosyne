"use client"

import { cn } from "@/lib/utils"

type Phase = "setup" | "running" | "halted"

interface PhaseStepperProps {
  phase: Phase
}

const STEPS: { id: Phase; label: string }[] = [
  { id: "setup", label: "SETUP" },
  { id: "running", label: "RUNNING" },
  { id: "halted", label: "HALTED" },
]

export function PhaseStepper({ phase }: PhaseStepperProps) {
  const currentIndex = STEPS.findIndex((s) => s.id === phase)

  return (
    <div className="flex items-center gap-1.5 text-[0.65rem] tracking-[0.2em] uppercase">
      {STEPS.map((step, i) => (
        <span key={step.id} className="flex items-center gap-1.5">
          <span
            className={cn(
              "text-muted-foreground",
              i === currentIndex && "text-accent opacity-100",
              i < currentIndex && "opacity-60",
              i > currentIndex && "opacity-30",
            )}
          >
            {step.label}
          </span>
          {i < STEPS.length - 1 && <span className="text-border">/</span>}
        </span>
      ))}
    </div>
  )
}
