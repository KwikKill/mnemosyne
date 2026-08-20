"use client"

import { cn } from "@/lib/utils"

type Phase = "setup" | "running" | "halted"

interface StatusChipProps {
  phase: Phase
  haltReason?: "loss" | "steps" | "abort"
}

export function StatusChip({ phase, haltReason }: StatusChipProps) {
  const label =
    phase === "setup" ? "STANDBY" : phase === "running" ? "TRAINING" : haltReason === "abort" ? "ABORTED" : "HALTED"
  const tone = phase === "running" ? "text-accent" : haltReason === "abort" ? "text-alert" : "text-muted-foreground"

  return (
    <div className="flex items-center gap-1.5 border border-border px-2 py-1">
      <span
        className={cn(
          "size-1.5 rounded-full",
          phase === "running" ? "bg-accent animate-[mnemo-caret_1s_steps(1)_infinite]" : "bg-muted-foreground",
        )}
      />
      <span className={cn("text-[0.65rem] tracking-[0.2em] uppercase", tone)}>STATUS: {label}</span>
    </div>
  )
}
