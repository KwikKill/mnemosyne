"use client"

import { type FormEvent, useRef, useState } from "react"
import type { TrainingEngine } from "@/lib/engine"
import { cn } from "@/lib/utils"

interface CommandConsoleProps {
  engine: TrainingEngine
}

interface ConsoleLine {
  id: number
  text: string
  tone: "input" | "output" | "error"
}

const HELP_TEXT = [
  "HELP",
  "SET LR <value>",
  "SET SPEED <value>",
  "RUN | PAUSE | ABORT",
  "PROBE <x> <y>",
  "PRUNE <fraction 0-1>",
  "CLEAR",
]

function argmax(row: number[]): number {
  let best = 0
  for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k
  return best
}

let idCounter = 0

function execute(engine: TrainingEngine, raw: string): string[] {
  const parts = raw.trim().split(/\s+/)
  const cmd = (parts[0] ?? "").toUpperCase()

  if (cmd === "") return []
  if (cmd === "HELP") return HELP_TEXT

  if (cmd === "SET" && parts[1]?.toUpperCase() === "LR") {
    const value = Number(parts[2])
    if (Number.isNaN(value)) return ["ERR: SET LR expects a number"]
    engine.setConfig({ learningRate: value })
    return [`LR set to ${value}`]
  }
  if (cmd === "SET" && parts[1]?.toUpperCase() === "SPEED") {
    const value = Number(parts[2])
    if (Number.isNaN(value)) return ["ERR: SET SPEED expects a number"]
    engine.setConfig({ stepsPerFrame: value })
    return [`Sim speed set to ${value}x`]
  }
  if (cmd === "RUN" || cmd === "PLAY") {
    engine.play()
    return ["sequence resumed"]
  }
  if (cmd === "PAUSE") {
    engine.pause()
    return ["sequence paused"]
  }
  if (cmd === "ABORT") {
    engine.abort()
    return ["sequence aborted"]
  }
  if (cmd === "PROBE") {
    const x = Number(parts[1])
    const y = Number(parts[2])
    if (Number.isNaN(x) || Number.isNaN(y)) return ["ERR: PROBE expects <x> <y>"]
    const [row] = engine.network.predict([[x, y]])
    const isBinary = engine.dataset.numClasses === 2
    const predictedClass = isBinary ? (row[0] >= 0.5 ? 1 : 0) : argmax(row)
    const confidence = isBinary ? (predictedClass === 1 ? row[0] : 1 - row[0]) : row[predictedClass]
    return [`CLASS ${predictedClass} · ${(confidence * 100).toFixed(1)}%`]
  }
  if (cmd === "PRUNE") {
    const fraction = Number(parts[1])
    if (Number.isNaN(fraction) || fraction <= 0 || fraction >= 1) return ["ERR: PRUNE expects a fraction between 0 and 1"]
    engine.prune(fraction)
    return [`pruning ${(fraction * 100).toFixed(0)}% of synapses`]
  }
  if (cmd === "CLEAR") return ["__CLEAR__"]

  return [`ERR: unknown command "${cmd}". Type HELP.`]
}

export function CommandConsole({ engine }: CommandConsoleProps) {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState("")
  const [lines, setLines] = useState<ConsoleLine[]>([
    { id: idCounter++, text: "COMMAND INTERFACE ONLINE. TYPE HELP.", tone: "output" },
  ])
  const scrollRef = useRef<HTMLDivElement>(null)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const raw = input
    setInput("")
    if (!raw.trim()) return
    const results = execute(engine, raw)
    if (results[0] === "__CLEAR__") {
      setLines([])
      return
    }
    setLines((prev) => [
      ...prev,
      { id: idCounter++, text: raw, tone: "input" as const },
      ...results.map((text) => ({ id: idCounter++, text, tone: (text.startsWith("ERR") ? "error" : "output") as ConsoleLine["tone"] })),
    ].slice(-100))
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight }))
  }

  return (
    <div className="fixed bottom-4 right-4 z-60 flex flex-col items-end gap-2">
      {open && (
        <div className="mnemo-panel w-80 max-w-[calc(100vw-2rem)] flex flex-col gap-2 p-3">
          <div className="mnemo-header">Command Interface</div>
          <div ref={scrollRef} className="h-48 overflow-y-auto text-xs leading-relaxed space-y-0.5 pr-1 font-mono">
            {lines.map((line) => (
              <p
                key={line.id}
                className={cn(
                  line.tone === "input" && "text-foreground",
                  line.tone === "output" && "text-muted-foreground",
                  line.tone === "error" && "text-alert",
                )}
              >
                {line.tone === "input" ? `> ${line.text}` : line.text}
              </p>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="flex gap-1.5">
            <span className="text-accent text-xs pt-1.5">{">"}</span>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="HELP"
              className="flex-1 bg-transparent border border-border px-2 py-1 text-xs text-foreground focus:outline-none focus:border-accent/60"
              autoFocus
            />
          </form>
        </div>
      )}
      <button
        onClick={() => setOpen((v) => !v)}
        className="size-10 border border-accent bg-panel text-accent flex items-center justify-center text-sm cursor-pointer hover:bg-accent/15"
        aria-label="Toggle command console"
      >
        {open ? "×" : ">_"}
      </button>
    </div>
  )
}
