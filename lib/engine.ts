// Main-thread shim: owns a Worker running the real training loop
// (lib/training-worker.ts) and republishes what it posts back through the
// same subscribe()/EngineTick contract this class always exposed, so
// components that only ever talked to it that way (run-status.tsx,
// system-log.tsx, loss-chart.tsx) need no changes for the Worker move.
// What does change: `network` is now a read-only NetworkSnapshot mirrored
// from the worker's postMessage replies, not a live class instance, since
// the real Network only exists on the worker's side of the thread boundary.
import { forwardPass, type ClassificationMetrics, type LayerLike, type NetworkSnapshot } from "./nn/network"
import type { Dataset } from "./nn/datasets"
import type { EngineConfig, EngineEvent, HistorySnapshot, TickPayload, WorkerCommand, WorkerMessage } from "./worker-protocol"

export type { EngineConfig, EngineEvent, HistorySnapshot, StopCondition } from "./worker-protocol"

// Turns a stored replay point back into the same NetworkSnapshot shape the
// live `engine.network` already is, so network-diagram.tsx/
// decision-boundary.tsx can render either without knowing which they got.
export function networkSnapshotFromHistory(snap: HistorySnapshot): NetworkSnapshot {
  const layers = snap.layers
  const sizes = [layers[0].weights.length, ...layers.map((l) => l.weights[0].length)]
  return {
    sizes,
    layers,
    lastActivations: null,
    predict: (X) => forwardPass(layers, X).at(-1)!,
  }
}

export interface EngineTick {
  step: number
  loss: number
  valLoss?: number
  effectiveLearningRate: number
  network: NetworkSnapshot
  dataset: Dataset
  events: EngineEvent[]
}

type Listener = (tick: EngineTick) => void

function snapshotFromTick(tick: TickPayload): NetworkSnapshot {
  const layers: LayerLike[] = tick.layers
  return {
    sizes: tick.sizes,
    layers,
    lastActivations: tick.lastActivations,
    predict: (X) => forwardPass(layers, X).at(-1)!,
  }
}

// Placeholder read before the worker's first reply arrives (postMessage is
// always asynchronous, even to a worker spun up in the same tick). Every
// consumer that reads engine.network/engine.dataset synchronously on mount
// (network-diagram.tsx, decision-boundary.tsx) already tolerates an empty
// network safely, they just draw nothing until the first real tick lands.
const EMPTY_NETWORK: NetworkSnapshot = {
  sizes: [],
  layers: [],
  lastActivations: null,
  predict: (X) => X.map(() => [0]),
}
const EMPTY_DATASET: Dataset = { id: "xor", label: "", X: [], Y: [], numClasses: 2 }

export class TrainingEngine {
  network: NetworkSnapshot = EMPTY_NETWORK
  dataset: Dataset = EMPTY_DATASET
  config: EngineConfig
  step = 0
  playing = false
  halted = false
  // The decimated weight-history timeline for the current run, populated
  // once on the tick that carries the "halted" event. Null until then, and
  // reset to null on every fresh reset() (see handleMessage's "reset").
  history: HistorySnapshot[] | null = null
  // Same reasoning as `history`: a plain property set synchronously in
  // handleMessage, read at mount by result-panel.tsx's useState initializer.
  // ResultPanel only starts existing on the same tick that carries this
  // event (it's what makes phase flip to "halted" in the first place), so
  // subscribing to catch the event live would always be one tick too late,
  // by the time its effect runs, that tick has already been dispatched to
  // whichever listeners existed *before* ResultPanel mounted.
  metrics: ClassificationMetrics | null = null
  private worker: Worker | null = null
  private listeners = new Set<Listener>()

  constructor(config: EngineConfig) {
    this.config = config
    // Next prerenders this app's route statically, which runs this
    // constructor once in Node during `next build` (no Worker/window
    // there). Every real use happens client-side after hydration.
    if (typeof window !== "undefined") {
      this.worker = new Worker(new URL("./training-worker.ts", import.meta.url), { type: "module" })
      this.worker.onmessage = (e: MessageEvent<WorkerMessage>) => this.handleMessage(e.data)
      this.send({ type: "init", config })
    }
  }

  private send(command: WorkerCommand) {
    this.worker?.postMessage(command)
  }

  private handleMessage(message: WorkerMessage) {
    if (message.type === "dataset") {
      this.dataset = message.dataset
      return
    }
    const { tick } = message
    this.step = tick.step
    this.network = snapshotFromTick(tick)
    for (const event of tick.events) {
      if (event.type === "reset") {
        this.halted = false
        this.history = null
        this.metrics = null
      }
      if (event.type === "halted") {
        this.halted = true
        this.playing = false
      }
      if (event.type === "metrics") this.metrics = event.metrics
    }
    if (tick.history) this.history = tick.history
    for (const fn of this.listeners) {
      fn({
        step: tick.step,
        loss: tick.loss,
        valLoss: tick.valLoss,
        effectiveLearningRate: tick.effectiveLearningRate,
        network: this.network,
        dataset: this.dataset,
        events: tick.events,
      })
    }
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  play() {
    if (this.playing || this.halted) return
    this.playing = true
    this.send({ type: "play" })
  }

  pause() {
    this.playing = false
    this.send({ type: "pause" })
  }

  stepOnce() {
    if (this.halted) return
    this.playing = false
    this.send({ type: "step" })
  }

  abort() {
    if (this.halted) return
    this.playing = false
    this.send({ type: "abort" })
  }

  // Zeroes the smallest-magnitude weights across the network and reports
  // the loss impact via a "pruned" event on the next tick. Safe to call
  // any time there's a trained network worth pruning, the result panel
  // only exposes it once halted but nothing here requires that.
  prune(fraction: number) {
    this.send({ type: "prune", fraction })
  }

  reset(config: EngineConfig = this.config) {
    this.config = config
    this.playing = false
    this.halted = false
    this.send({ type: "reset", config })
  }

  // Learning rate / speed / stop condition are safe to hot-swap; anything
  // that changes the network's shape or the dataset itself (architecture,
  // activation, dataset id, noise, custom points) requires a reset, the
  // worker makes that same decision on its side, this only keeps the
  // main-thread mirror of `config` current for synchronous reads
  // (run-status.tsx's stopLabel, for instance).
  setConfig(partial: Partial<EngineConfig>) {
    this.config = { ...this.config, ...partial }
    this.send({ type: "setConfig", partial })
  }

  destroy() {
    this.worker?.terminate()
    this.worker = null
    this.listeners.clear()
  }
}
