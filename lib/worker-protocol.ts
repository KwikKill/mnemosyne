// Shared message shapes between the main thread (lib/engine.ts) and the
// training worker (lib/training-worker.ts). Kept in one place so a change
// on one side of postMessage shows up as a type error on the other.
import type { CustomPoint, DatasetId } from "./nn/datasets"
import type { LrSchedule } from "./lr-schedule"
import type { Activation, ClassificationMetrics, LayerLike } from "./nn/network"

export type StopCondition = { type: "loss"; target: number } | { type: "steps"; target: number } | { type: "manual" }

export interface EngineConfig {
  datasetId: DatasetId
  hiddenLayers: number[]
  learningRate: number
  activation: Activation
  stepsPerFrame: number
  stopCondition: StopCondition
  lrSchedule: LrSchedule
  noise?: number
  // Only read when datasetId === "custom"; the dataset painter's points and
  // the class count the operator chose for them.
  customPoints?: CustomPoint[]
  customNumClasses?: number
}

export type EngineEvent =
  | { type: "reset" }
  | { type: "dataset-change"; datasetId: DatasetId }
  | { type: "milestone"; threshold: number }
  | { type: "halted"; reason: "loss" | "steps" | "abort"; step: number; loss: number }
  | { type: "overfitting"; step: number }
  | { type: "pruned"; fraction: number; lossBefore: number; lossAfter: number }
  | { type: "metrics"; metrics: ClassificationMetrics }

// Commands: main thread -> worker.
export type WorkerCommand =
  | { type: "init"; config: EngineConfig }
  | { type: "play" }
  | { type: "pause" }
  | { type: "step" }
  | { type: "abort" }
  | { type: "reset"; config: EngineConfig }
  | { type: "setConfig"; partial: Partial<EngineConfig> }
  | { type: "prune"; fraction: number }

// A single point along a training run's timeline, kept for the replay
// scrubber. Deliberately just the weights, not activations or anything
// else, that's all a historical decision-boundary/network-diagram redraw
// needs.
export interface HistorySnapshot {
  step: number
  loss: number
  layers: LayerLike[]
}

// A serialized tick, worker -> main thread. Plain data only, everything
// here survives structured clone across postMessage (no class instances,
// no functions).
export interface TickPayload {
  step: number
  loss: number
  valLoss?: number
  effectiveLearningRate: number
  sizes: number[]
  layers: LayerLike[]
  lastActivations: number[][][] | null
  events: EngineEvent[]
  // Only present on the tick that carries the "halted" event, the full
  // decimated timeline for that run.
  history?: HistorySnapshot[]
}

export interface DatasetPayload {
  id: DatasetId
  label: string
  X: number[][]
  Y: number[][]
  numClasses: number
}

export type WorkerMessage = { type: "tick"; tick: TickPayload } | { type: "dataset"; dataset: DatasetPayload }
