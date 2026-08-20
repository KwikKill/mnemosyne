// Pure learning-rate schedule math, shared by the worker (which actually
// trains with it) and the setup screen (which only draws a preview curve).
// Keeping this in one place means the preview can never drift from what
// training actually does.
export type LrSchedule =
  | { type: "none" }
  | { type: "step"; every: number; factor: number }
  | { type: "cosine"; totalSteps: number }

export function computeEffectiveLR(schedule: LrSchedule, baseLR: number, step: number): number {
  switch (schedule.type) {
    case "none":
      return baseLR
    case "step": {
      const drops = Math.floor(step / schedule.every)
      return baseLR * schedule.factor ** drops
    }
    case "cosine": {
      const t = Math.min(step, schedule.totalSteps) / schedule.totalSteps
      return baseLR * 0.5 * (1 + Math.cos(Math.PI * t))
    }
  }
}
