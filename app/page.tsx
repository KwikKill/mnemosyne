"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { BootSequence } from "@/components/boot-sequence"
import { CommandConsole } from "@/components/command-console"
import { CrtOverlay } from "@/components/crt-overlay"
import { DecisionBoundary } from "@/components/decision-boundary"
import { LossChart } from "@/components/loss-chart"
import { NetworkDiagram } from "@/components/network-diagram"
import { PhaseStepper } from "@/components/phase-stepper"
import { ResultPanel } from "@/components/result-panel"
import { RunStatus } from "@/components/run-status"
import { SetupScreen } from "@/components/setup-screen"
import { StatusChip } from "@/components/status-chip"
import { SystemLog } from "@/components/system-log"
import { WeightHistogram } from "@/components/weight-histogram"
import { type EngineConfig, networkSnapshotFromHistory, TrainingEngine } from "@/lib/engine"
import { DEFAULT_PROTOCOL } from "@/lib/protocol"

type Phase = "setup" | "running" | "halted"

interface HaltInfo {
  reason: "loss" | "steps" | "abort"
  step: number
  loss: number
}

const DEFAULT_CONFIG: EngineConfig = {
  ...DEFAULT_PROTOCOL.config,
  stopCondition: DEFAULT_PROTOCOL.stopCondition,
}

export default function Home() {
  const engineRef = useRef<TrainingEngine | null>(null)
  if (!engineRef.current) engineRef.current = new TrainingEngine(DEFAULT_CONFIG)
  const engine = engineRef.current

  const [booted, setBooted] = useState(false)
  const [phase, setPhase] = useState<Phase>("setup")
  const [config, setConfig] = useState<EngineConfig>(DEFAULT_CONFIG)
  const [haltInfo, setHaltInfo] = useState<HaltInfo | null>(null)
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null)
  const [replayIndex, setReplayIndex] = useState<number | null>(null)
  const [transitionKey, setTransitionKey] = useState(0)
  const historyPoint = replayIndex !== null ? (engine.history?.[replayIndex] ?? null) : null
  const overrideNetwork = useMemo(() => (historyPoint ? networkSnapshotFromHistory(historyPoint) : null), [historyPoint])
  const startPendingRef = useRef(false)
  const isFirstPhaseRef = useRef(true)

  // Brief screen-flash whenever the phase actually changes, not on first
  // mount (that moment belongs to the boot sequence, not a "transition").
  useEffect(() => {
    if (isFirstPhaseRef.current) {
      isFirstPhaseRef.current = false
      return
    }
    setTransitionKey((k) => k + 1)
  }, [phase])

  // Always-mounted, phase-independent: the one place the engine's own
  // "halted" event (a stop condition firing) turns into a phase change.
  // Explicit user actions (INITIATE/RUN AGAIN/ABORT) set `phase` directly
  // elsewhere; this only reacts to the engine reaching a halt on its own.
  useEffect(() => {
    return engine.subscribe((tick) => {
      for (const event of tick.events) {
        if (event.type === "halted") {
          setHaltInfo({ reason: event.reason, step: event.step, loss: event.loss })
          setPhase("halted")
        }
      }
    })
  }, [engine])

  // Runs after the dashboard's own components have mounted and subscribed
  // (child effects fire before this parent effect in the same commit), so
  // the reset()/play() call below never emits into an empty listener set:
  // the system log's "NETWORK REINITIALIZED" line is never silently lost.
  useEffect(() => {
    if (phase === "running" && startPendingRef.current) {
      startPendingRef.current = false
      engine.reset(config)
      engine.play()
    }
  }, [phase, config, engine])

  useEffect(() => () => engine.destroy(), [engine])

  const startRun = (nextConfig: EngineConfig) => {
    setConfig(nextConfig)
    setRecordingUrl(null)
    setReplayIndex(null)
    startPendingRef.current = true
    setPhase("running")
  }

  const handleRunAgain = () => startRun(config)
  const handleReconfigure = () => setPhase("setup")
  const handleNewProtocol = () => {
    setConfig(DEFAULT_CONFIG)
    setPhase("setup")
  }

  if (!booted) {
    return <BootSequence onComplete={() => setBooted(true)} />
  }

  return (
    <main className="relative min-h-screen p-4 sm:p-8">
      <CrtOverlay engine={engine} />
      {transitionKey > 0 && (
        <div
          key={transitionKey}
          aria-hidden="true"
          className="pointer-events-none fixed inset-0 z-40 bg-accent/25"
          style={{ animation: "mnemo-phase-flash 0.35s ease-out forwards" }}
        />
      )}
      <div className="max-w-7xl mx-auto flex flex-col gap-4">
        <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border pb-3">
          <div>
            <h1 className="text-lg tracking-[0.3em] text-foreground animate-[mnemo-logo-settle_0.9s_ease-out_both]">
              MNEMOSYNE
            </h1>
            <p className="mnemo-label mt-1">Neural Diagnostic Terminal // Live Synaptic Trace</p>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <p className="mnemo-label hidden sm:block">STATION-7 // OBSERVATION DECK</p>
            <div className="flex items-center gap-3">
              <PhaseStepper phase={phase} />
              <StatusChip phase={phase} haltReason={haltInfo?.reason} />
            </div>
          </div>
        </header>

        {phase === "setup" ? (
          <SetupScreen initialConfig={config} onInitiate={startRun} />
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-[1.6fr_1.1fr_1fr] gap-4 h-[64rem] lg:h-[34rem]">
              <div className="mnemo-panel-in h-full min-h-0" style={{ animationDelay: "0ms" }}>
                <NetworkDiagram engine={engine} overrideNetwork={overrideNetwork} />
              </div>
              <div className="mnemo-panel-in h-full min-h-0" style={{ animationDelay: "60ms" }}>
                <DecisionBoundary
                  engine={engine}
                  probeEnabled={phase === "halted"}
                  recording={phase === "running"}
                  onRecordingReady={setRecordingUrl}
                  overrideNetwork={overrideNetwork}
                />
              </div>
              <div className="flex flex-col gap-4 h-full min-h-0">
                <div className="mnemo-panel-in flex-1 min-h-0" style={{ animationDelay: "120ms" }}>
                  <LossChart engine={engine} />
                </div>
                <div className="mnemo-panel-in flex-1 min-h-0" style={{ animationDelay: "180ms" }}>
                  <WeightHistogram engine={engine} />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-4 h-[30rem] lg:h-[28rem]">
              <div className="mnemo-panel-in h-full min-h-0" style={{ animationDelay: "240ms" }}>
                {phase === "running" ? (
                  <RunStatus engine={engine} />
                ) : (
                  haltInfo && (
                    <ResultPanel
                      engine={engine}
                      haltInfo={haltInfo}
                      recordingUrl={recordingUrl}
                      replayIndex={replayIndex}
                      onReplayChange={setReplayIndex}
                      onRunAgain={handleRunAgain}
                      onReconfigure={handleReconfigure}
                      onNewProtocol={handleNewProtocol}
                    />
                  )
                )}
              </div>
              <div className="mnemo-panel-in h-full min-h-0" style={{ animationDelay: "300ms" }}>
                <SystemLog engine={engine} />
              </div>
            </div>
            <CommandConsole engine={engine} />
          </>
        )}

        <footer className="mnemo-label text-center pb-2">
          MNEMOSYNE // hand-written MLP · &copy; {new Date().getFullYear()}{" "}
          <a
            href="https://gabriel.blaisot.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-foreground/70 underline decoration-accent/40 underline-offset-2 transition-colors hover:text-accent"
          >
            KwikKill
          </a>
        </footer>
      </div>
    </main>
  )
}
