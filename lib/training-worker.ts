/// <reference lib="webworker" />
// Runs the actual SGD loop off the main thread. Owns the real Network and
// Dataset instances; the main thread (lib/engine.ts) only ever sees the
// serialized snapshots posted back from here. Paced with setTimeout rather
// than requestAnimationFrame, since dedicated workers don't reliably get
// rAF, and it decouples training speed from the page's render cadence, a
// real advantage of running here rather than on the main thread.
import {
  buildCustomDataset,
  type Dataset,
  generateDataset,
  MIN_POINTS_FOR_VALIDATION,
  splitDataset,
} from "./nn/datasets"
import { computeEffectiveLR } from "./lr-schedule"
import { computeMetrics, Network } from "./nn/network"
import type { EngineConfig, EngineEvent, HistorySnapshot, WorkerCommand, WorkerMessage } from "./worker-protocol"

const MILESTONES = [0.5, 0.3, 0.15, 0.08, 0.04, 0.02]
const FRAME_MS = 16
const VAL_FRACTION = 0.2
// Overfitting fires once per run when validation loss sits 15% above its
// best-so-far for 5 consecutive ticks in a row, not on the first uptick,
// toy-dataset loss curves are naturally noisy enough to false-positive on
// a single bad tick.
const OVERFIT_MARGIN = 1.15
const OVERFIT_PATIENCE = 5
// Replay history stays bounded regardless of run length: snapshot every
// `snapshotInterval` steps, and once the count exceeds the cap, drop every
// other snapshot and double the interval, halving resolution instead of
// growing memory without limit.
const HISTORY_CAP = 200

let network: Network
let dataset: Dataset
let config: EngineConfig
let step = 0
let lastLoss = Number.POSITIVE_INFINITY
let lastValLoss: number | undefined
let halted = false
let playing = false
let crossedMilestones = new Set<number>()
let pendingEvents: EngineEvent[] = []
let timer: ReturnType<typeof setTimeout> | null = null

let trainX: number[][] = []
let trainY: number[][] = []
let valX: number[][] | null = null
let valY: number[][] | null = null
let minValLoss = Number.POSITIVE_INFINITY
let noImprovementStreak = 0
let overfitFlagged = false

let history: HistorySnapshot[] = []
let snapshotInterval = 10

function buildDataset(cfg: EngineConfig): Dataset {
  if (cfg.datasetId === "custom") {
    return buildCustomDataset(cfg.customPoints ?? [], cfg.customNumClasses ?? 2)
  }
  return generateDataset(cfg.datasetId, cfg.noise)
}

function postTick() {
  const events = pendingEvents
  pendingEvents = []
  const message: WorkerMessage = {
    type: "tick",
    tick: {
      step,
      loss: lastLoss,
      valLoss: lastValLoss,
      effectiveLearningRate: computeEffectiveLR(config.lrSchedule, config.learningRate, step),
      sizes: network.sizes,
      layers: network.layers,
      lastActivations: network.lastActivations,
      events,
      history: halted ? history : undefined,
    },
  }
  postMessage(message)
}

function cloneLayers(layers: Network["layers"]): Network["layers"] {
  return layers.map((l) => ({ weights: l.weights.map((row) => [...row]), biases: [...l.biases], activation: l.activation }))
}

function maybeSnapshot() {
  if (step % snapshotInterval !== 0) return
  history.push({ step, loss: lastLoss, layers: cloneLayers(network.layers) })
  if (history.length > HISTORY_CAP) {
    history = history.filter((_, i) => i % 2 === 0)
    snapshotInterval *= 2
  }
}

function postDataset() {
  const message: WorkerMessage = {
    type: "dataset",
    dataset: { id: dataset.id, label: dataset.label, X: dataset.X, Y: dataset.Y, numClasses: dataset.numClasses },
  }
  postMessage(message)
}

function checkOverfitting() {
  if (valX === null || lastValLoss === undefined || overfitFlagged) return
  if (lastValLoss < minValLoss) {
    minValLoss = lastValLoss
    noImprovementStreak = 0
    return
  }
  if (lastValLoss > minValLoss * OVERFIT_MARGIN) {
    noImprovementStreak++
    if (noImprovementStreak >= OVERFIT_PATIENCE) {
      overfitFlagged = true
      pendingEvents.push({ type: "overfitting", step })
    }
  } else {
    noImprovementStreak = 0
  }
}

function trainOnce() {
  const effectiveLR = computeEffectiveLR(config.lrSchedule, config.learningRate, step)
  const { loss } = network.trainStep(trainX, trainY, effectiveLR)
  step++
  lastLoss = loss
  maybeSnapshot()
  if (valX !== null && valY !== null) {
    lastValLoss = network.evaluate(valX, valY)
    checkOverfitting()
  }
  for (const threshold of MILESTONES) {
    if (loss < threshold && !crossedMilestones.has(threshold)) {
      crossedMilestones.add(threshold)
      pendingEvents.push({ type: "milestone", threshold })
    }
  }
  checkStopCondition()
}

// Shared terminal-state transition for every halt path (a stop condition
// firing or an operator abort): marks halted, announces why, and computes
// classification metrics once against whichever labeled set is actually
// meaningful (validation split if one exists, the training set otherwise).
function haltWith(reason: "loss" | "steps" | "abort") {
  halted = true
  pendingEvents.push({ type: "halted", reason, step, loss: lastLoss })
  const metricsX = valX ?? trainX
  const metricsY = valY ?? trainY
  pendingEvents.push({ type: "metrics", metrics: computeMetrics(network, metricsX, metricsY, network.numClasses) })
}

function checkStopCondition() {
  const sc = config.stopCondition
  let reason: "loss" | "steps" | null = null
  if (sc.type === "loss" && lastLoss <= sc.target) reason = "loss"
  else if (sc.type === "steps" && step >= sc.target) reason = "steps"
  if (!reason) return
  haltWith(reason)
}

function loop() {
  if (!playing) return
  for (let i = 0; i < config.stepsPerFrame && !halted; i++) trainOnce()
  postTick()
  if (halted) {
    playing = false
    timer = null
    return
  }
  timer = setTimeout(loop, FRAME_MS)
}

function play() {
  if (playing || halted) return
  playing = true
  timer = setTimeout(loop, FRAME_MS)
}

function pause() {
  playing = false
  if (timer !== null) clearTimeout(timer)
  timer = null
}

function stepOnce() {
  if (halted) return
  pause()
  trainOnce()
  postTick()
}

function abort() {
  if (halted) return
  pause()
  haltWith("abort")
  postTick()
}

function prune(fraction: number) {
  const lossBefore = network.evaluate(trainX, trainY)
  network.pruneWeakWeights(fraction)
  const lossAfter = network.evaluate(trainX, trainY)
  lastLoss = lossAfter
  pendingEvents.push({ type: "pruned", fraction, lossBefore, lossAfter })
  postTick()
}

function resetWith(cfg: EngineConfig) {
  pause()
  const datasetChanged = config === undefined || cfg.datasetId !== config.datasetId
  config = cfg
  dataset = buildDataset(cfg)

  if (dataset.X.length >= MIN_POINTS_FOR_VALIDATION) {
    const split = splitDataset(dataset, VAL_FRACTION)
    trainX = split.train.X
    trainY = split.train.Y
    valX = split.val.X
    valY = split.val.Y
  } else {
    trainX = dataset.X
    trainY = dataset.Y
    valX = null
    valY = null
  }
  minValLoss = Number.POSITIVE_INFINITY
  noImprovementStreak = 0
  overfitFlagged = false
  lastValLoss = undefined
  history = []
  snapshotInterval = 10

  // Network's own convention: pass 1 for binary (single sigmoid unit),
  // the real class count only for 3+ classes (softmax layer).
  const outputSize = dataset.numClasses === 2 ? 1 : dataset.numClasses
  network = new Network([2, ...cfg.hiddenLayers, outputSize], cfg.activation)
  step = 0
  lastLoss = Number.POSITIVE_INFINITY
  halted = false
  crossedMilestones = new Set()
  pendingEvents = [{ type: "reset" }]
  if (datasetChanged) pendingEvents.push({ type: "dataset-change", datasetId: cfg.datasetId })
  postDataset()
  postTick()
}

self.onmessage = (e: MessageEvent<WorkerCommand>) => {
  const command = e.data
  switch (command.type) {
    case "init":
      resetWith(command.config)
      break
    case "play":
      play()
      break
    case "pause":
      pause()
      break
    case "step":
      stepOnce()
      break
    case "abort":
      abort()
      break
    case "reset":
      resetWith(command.config)
      break
    case "prune":
      prune(command.fraction)
      break
    case "setConfig": {
      const partial = command.partial
      const needsReset =
        "hiddenLayers" in partial ||
        "activation" in partial ||
        "datasetId" in partial ||
        "noise" in partial ||
        "customPoints" in partial ||
        "customNumClasses" in partial
      const next = { ...config, ...partial }
      if (needsReset) {
        resetWith(next)
      } else {
        config = next
      }
      break
    }
  }
}
