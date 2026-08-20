"use client"

import { useEffect, useState } from "react"
import { ClassificationMetricsTable } from "@/components/classification-metrics"
import { Slider } from "@/components/field-controls"
import { ReplayScrubber } from "@/components/replay-scrubber"
import type { EngineTick, HistorySnapshot, TrainingEngine } from "@/lib/engine"
import type { ClassificationMetrics } from "@/lib/nn/network"

interface HaltInfo {
  reason: "loss" | "steps" | "abort"
  step: number
  loss: number
}

interface PruneResult {
  fraction: number
  lossBefore: number
  lossAfter: number
}

interface ResultPanelProps {
  engine: TrainingEngine
  haltInfo: HaltInfo
  recordingUrl: string | null
  replayIndex: number | null
  onReplayChange: (index: number | null) => void
  onRunAgain: () => void
  onReconfigure: () => void
  onNewProtocol: () => void
}

const REASON_TEXT: Record<HaltInfo["reason"], string> = {
  loss: "CONVERGENCE TARGET REACHED",
  steps: "STEP LIMIT REACHED",
  abort: "SEQUENCE ABORTED BY OPERATOR",
}

export function ResultPanel({
  engine,
  haltInfo,
  recordingUrl,
  replayIndex,
  onReplayChange,
  onRunAgain,
  onReconfigure,
  onNewProtocol,
}: ResultPanelProps) {
  const { config } = engine
  const [pruneFraction, setPruneFraction] = useState(0.2)
  const [pruneResult, setPruneResult] = useState<PruneResult | null>(null)
  // Both read from the engine's own plain properties, not caught live via
  // subscribe(): ResultPanel only starts existing on the same tick that
  // carries the "halted"/"metrics" events (that's what makes phase flip to
  // "halted"), so by the time its effect below subscribes, that tick has
  // already been dispatched to whichever listeners existed before this
  // component mounted. Reading engine.metrics/.history at mount works
  // because TrainingEngine.handleMessage sets them *before* notifying
  // listeners.
  const [metrics, setMetrics] = useState<ClassificationMetrics | null>(engine.metrics)
  const [history, setHistory] = useState<HistorySnapshot[] | null>(engine.history)

  useEffect(() => {
    return engine.subscribe((tick: EngineTick) => {
      for (const event of tick.events) {
        if (event.type === "pruned") {
          setPruneResult({ fraction: event.fraction, lossBefore: event.lossBefore, lossAfter: event.lossAfter })
        }
      }
      if (engine.metrics) setMetrics(engine.metrics)
      if (engine.history) setHistory(engine.history)
    })
  }, [engine])

  return (
    <div className="mnemo-panel h-full p-3 flex flex-col gap-4 overflow-y-auto">
      <div className="mnemo-header">Protocol Complete</div>

      <div className="flex flex-col gap-1">
        <div className="text-accent text-xs tracking-wider">{REASON_TEXT[haltInfo.reason]}</div>
        <div className="text-muted-foreground text-xs">
          {haltInfo.step} steps · final loss {haltInfo.loss.toFixed(4)}
        </div>
        <div className="text-muted-foreground text-xs">
          {config.datasetId.toUpperCase()} · [{config.hiddenLayers.join(", ")}] · {config.activation.toUpperCase()}
        </div>
      </div>

      <p className="mnemo-label normal-case tracking-normal text-[0.65rem] leading-snug border-t border-border pt-3">
        Click the classification surface to probe the trained network at any point.
      </p>

      <div className="flex flex-col gap-2 border-t border-border pt-3">
        <div className="mnemo-label">Synaptic Pruning</div>
        <Slider
          label="Fraction"
          value={pruneFraction}
          display={`${(pruneFraction * 100).toFixed(0)}%`}
          min={0.05}
          max={0.6}
          step={0.05}
          onChange={setPruneFraction}
        />
        <button
          onClick={() => engine.prune(pruneFraction)}
          className="border border-border text-muted-foreground px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:border-accent/50 hover:text-foreground"
        >
          PRUNE WEAK SYNAPSES
        </button>
        {pruneResult && (
          <p className="mnemo-label normal-case tracking-normal text-[0.65rem] leading-snug">
            {(pruneResult.fraction * 100).toFixed(0)}% zeroed. loss {pruneResult.lossBefore.toFixed(4)} →{" "}
            {pruneResult.lossAfter.toFixed(4)}
          </p>
        )}
      </div>

      {metrics && <ClassificationMetricsTable metrics={metrics} />}

      <ReplayScrubber history={history} index={replayIndex} onChange={onReplayChange} />

      {recordingUrl && (
        <a
          href={recordingUrl}
          download="mnemosyne-convergence.webm"
          className="border border-border text-muted-foreground px-2 py-1.5 text-xs tracking-widest text-center cursor-pointer hover:border-accent/50 hover:text-foreground"
        >
          DOWNLOAD CONVERGENCE CLIP (.WEBM)
        </a>
      )}

      <div className="flex flex-col gap-1.5 mt-auto pt-2">
        <button
          onClick={onRunAgain}
          className="border border-accent bg-accent/15 text-accent px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:bg-accent/25"
        >
          RUN AGAIN
        </button>
        <button
          onClick={onReconfigure}
          className="border border-border text-muted-foreground px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:border-accent/50 hover:text-foreground"
        >
          RECONFIGURE
        </button>
        <button
          onClick={onNewProtocol}
          className="border border-border text-muted-foreground px-2 py-1.5 text-xs tracking-widest cursor-pointer hover:border-accent/50 hover:text-foreground"
        >
          NEW PROTOCOL
        </button>
      </div>
    </div>
  )
}
