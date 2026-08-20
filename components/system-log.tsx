"use client"

import { useEffect, useRef, useState } from "react"
import type { EngineTick, TrainingEngine } from "@/lib/engine"
import type { NetworkSnapshot } from "@/lib/nn/network"

interface SystemLogProps {
  engine: TrainingEngine
}

interface LogLine {
  id: number
  text: string
  tone: "info" | "notice" | "alert"
}

const DATASET_NAMES: Record<string, string> = {
  xor: "QUADRANT DISPARITY",
  circles: "CONCENTRIC RINGS",
  spirals: "TWIN SPIRAL",
  triple: "TRIPLE CLUSTER",
  custom: "CUSTOM SPECIMEN",
}

const PING_INTERVAL = 320

const HALT_TEXT: Record<"loss" | "steps" | "abort", string> = {
  loss: "CONVERGENCE TARGET REACHED. PROTOCOL COMPLETE.",
  steps: "STEP LIMIT REACHED. PROTOCOL COMPLETE.",
  abort: "SEQUENCE ABORTED BY OPERATOR.",
}

function stamp(step: number) {
  return `T+${String(step).padStart(6, "0")}`
}

// Pulls one real weight out of the live network to ground the periodic
// diagnostic line in an actual number rather than fabricated flavor text.
function sampleSynapse(network: NetworkSnapshot): string {
  const layerIdx = Math.floor(Math.random() * network.layers.length)
  const layer = network.layers[layerIdx]
  const i = Math.floor(Math.random() * layer.weights.length)
  const j = Math.floor(Math.random() * layer.weights[0].length)
  const w = layer.weights[i][j]
  return `L${layerIdx + 1}[${i}→${j}] ${w >= 0 ? "+" : ""}${w.toFixed(3)}`
}

let idCounter = 0

export function SystemLog({ engine }: SystemLogProps) {
  const [lines, setLines] = useState<LogLine[]>([{ id: idCounter++, text: "TERMINAL ONLINE. AWAITING TRAINING SIGNAL.", tone: "info" }])
  const lastPingRef = useRef(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onTick = (tick: EngineTick) => {
      const next: LogLine[] = []
      for (const event of tick.events) {
        if (event.type === "reset") {
          next.push({ id: idCounter++, text: "[ NETWORK REINITIALIZED. WEIGHTS RESEEDED. ]", tone: "notice" })
        } else if (event.type === "dataset-change") {
          const name = DATASET_NAMES[event.datasetId] ?? event.datasetId.toUpperCase()
          next.push({ id: idCounter++, text: `SPECIMEN SET SWAPPED: ${name}`, tone: "notice" })
        } else if (event.type === "milestone") {
          next.push({
            id: idCounter++,
            text: `${stamp(tick.step)} CONVERGENCE THRESHOLD Δ<${event.threshold.toFixed(2)} REACHED`,
            tone: event.threshold <= 0.08 ? "notice" : "info",
          })
        } else if (event.type === "halted") {
          next.push({
            id: idCounter++,
            text: `${stamp(event.step)} [ ${HALT_TEXT[event.reason]} ]`,
            tone: event.reason === "abort" ? "alert" : "notice",
          })
        } else if (event.type === "overfitting") {
          next.push({
            id: idCounter++,
            text: `${stamp(event.step)} [ OVERFITTING DETECTED. VALIDATION DIVERGING FROM TRAIN. ]`,
            tone: "alert",
          })
        } else if (event.type === "pruned") {
          const delta = event.lossAfter - event.lossBefore
          next.push({
            id: idCounter++,
            text: `${stamp(tick.step)} SYNAPTIC PRUNE ${(event.fraction * 100).toFixed(0)}% | LOSS ${event.lossBefore.toFixed(4)} → ${event.lossAfter.toFixed(4)} (${delta >= 0 ? "+" : ""}${delta.toFixed(4)})`,
            tone: "notice",
          })
        }
      }
      if (tick.step - lastPingRef.current >= PING_INTERVAL) {
        lastPingRef.current = tick.step
        next.push({
          id: idCounter++,
          text: `${stamp(tick.step)} SYNAPSE SCAN ${sampleSynapse(tick.network)} | LOSS ${tick.loss.toFixed(4)}`,
          tone: "info",
        })
      }
      if (next.length === 0) return
      setLines((prev) => [...prev, ...next].slice(-80))
    }
    return engine.subscribe(onTick)
  }, [engine])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [lines])

  return (
    <div className="mnemo-panel h-full min-h-32 p-3 flex flex-col gap-2">
      <div className="mnemo-header">System Log</div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto text-xs leading-relaxed space-y-0.5 pr-1">
        {lines.map((line) => (
          <p
            key={line.id}
            className={
              line.tone === "alert"
                ? "text-alert"
                : line.tone === "notice"
                  ? "text-accent"
                  : "text-muted-foreground"
            }
          >
            {line.text}
          </p>
        ))}
        <span className="inline-block w-2 h-3 bg-accent align-middle animate-[mnemo-caret_1s_steps(1)_infinite]" />
      </div>
    </div>
  )
}
