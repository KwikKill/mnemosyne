"use client"

import { useEffect, useRef, useState } from "react"
import { Slider } from "@/components/field-controls"
import type { EngineTick, TrainingEngine } from "@/lib/engine"

interface RunStatusProps {
  engine: TrainingEngine
}

function stopLabel(engine: TrainingEngine): string {
  const sc = engine.config.stopCondition
  if (sc.type === "loss") return `LOSS < ${sc.target}`
  if (sc.type === "steps") return `${sc.target} STEPS`
  return "MANUAL"
}

// Loss starts near ln(2) ≈ 0.69 for a balanced binary classifier at
// initialization, used as the "0% progress" anchor for a loss-target halt,
// since there's no other principled upper bound to measure against.
const INITIAL_LOSS_ESTIMATE = 0.69

export function RunStatus({ engine }: RunStatusProps) {
  const [playing, setPlaying] = useState(true)
  const [learningRate, setLearningRate] = useState(engine.config.learningRate)
  const [speed, setSpeed] = useState(engine.config.stepsPerFrame)
  const stepRef = useRef<HTMLSpanElement>(null)
  const lossRef = useRef<HTMLSpanElement>(null)
  const lrRef = useRef<HTMLSpanElement>(null)
  const barRef = useRef<HTMLDivElement>(null)
  const scheduleActive = engine.config.lrSchedule.type !== "none"

  useEffect(() => {
    const onTick = (tick: EngineTick) => {
      if (stepRef.current) stepRef.current.textContent = String(tick.step)
      if (lossRef.current) lossRef.current.textContent = tick.loss.toFixed(4)
      if (lrRef.current) lrRef.current.textContent = tick.effectiveLearningRate.toFixed(3)
      if (barRef.current) {
        const sc = engine.config.stopCondition
        let fraction = 0
        if (sc.type === "steps") {
          fraction = Math.min(1, tick.step / sc.target)
        } else if (sc.type === "loss") {
          fraction = Math.min(1, Math.max(0, (INITIAL_LOSS_ESTIMATE - tick.loss) / (INITIAL_LOSS_ESTIMATE - sc.target)))
        }
        barRef.current.style.width = `${fraction * 100}%`
      }
    }
    return engine.subscribe(onTick)
  }, [engine])

  const togglePlaying = () => {
    const next = !playing
    setPlaying(next)
    if (next) engine.play()
    else engine.pause()
  }

  return (
    <div className="mnemo-panel h-full p-3 flex flex-col gap-4 overflow-y-auto">
      <div className="mnemo-header">
        <span>Run Status</span>
        <span>HALT: {stopLabel(engine)}</span>
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>
          STEP <span ref={stepRef} className="text-accent">0</span>
        </span>
        <span>
          LOSS <span ref={lossRef} className="text-accent">N/A</span>
        </span>
        {scheduleActive && (
          <span>
            LR <span ref={lrRef} className="text-accent">{engine.config.learningRate.toFixed(3)}</span>
          </span>
        )}
      </div>
      <div className="h-1 bg-muted relative overflow-hidden">
        <div ref={barRef} className="h-full bg-accent absolute left-0 top-0" style={{ width: "0%" }} />
      </div>

      <Slider
        label="Learning Rate"
        value={learningRate}
        display={learningRate.toFixed(2)}
        min={0.01}
        max={1}
        step={0.01}
        onChange={(v) => {
          setLearningRate(v)
          engine.setConfig({ learningRate: v })
        }}
      />
      <Slider
        label="Sim Speed"
        value={speed}
        display={`${speed}x`}
        min={1}
        max={40}
        step={1}
        onChange={(v) => {
          setSpeed(v)
          engine.setConfig({ stepsPerFrame: v })
        }}
      />

      <div className="flex gap-1.5 mt-auto pt-2">
        <button
          onClick={togglePlaying}
          className="flex-1 border border-accent bg-accent/15 text-accent px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:bg-accent/25"
        >
          {playing ? "PAUSE" : "RESUME"}
        </button>
        <button
          onClick={() => engine.abort()}
          className="flex-1 border border-alert/60 text-alert px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:bg-alert/15"
        >
          ABORT
        </button>
      </div>
    </div>
  )
}
