// A hand-written multilayer perceptron: explicit forward pass, explicit
// backward pass via the chain rule, explicit gradient-descent update. No
// autograd, no tensor library. Every derivative below is written out by
// hand so the math is inspectable, not just callable.
import {
  addBias,
  elementwise,
  map,
  type Matrix,
  multiply,
  randomMatrix,
  scale,
  softmaxRows,
  sumColumns,
  transpose,
} from "./matrix"

export type Activation = "relu" | "tanh" | "sigmoid"
// The output layer may also be "softmax" (multi-class), never a hidden
// layer, hidden layers only ever use one of the three above.
export type LayerActivation = Activation | "softmax"

const ACTIVATIONS: Record<Activation, { fn: (x: number) => number; derivative: (activated: number) => number }> = {
  relu: {
    fn: (x) => Math.max(0, x),
    derivative: (activated) => (activated > 0 ? 1 : 0),
  },
  tanh: {
    fn: (x) => Math.tanh(x),
    derivative: (activated) => 1 - activated * activated,
  },
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (activated) => activated * (1 - activated),
  },
}

export interface LayerLike {
  weights: Matrix // inputSize x outputSize
  biases: number[] // length outputSize
  activation: LayerActivation
}

type Layer = LayerLike

export interface ForwardCache {
  // activations[0] is the input; activations[i] is layer i's output after
  // its activation function, kept around because both backprop and the
  // live "neuron activation" visualization need it.
  activations: Matrix[]
}

export interface TrainResult {
  loss: number
  cache: ForwardCache
}

// The subset of Network that visualization code actually reads. A real
// Network satisfies this structurally; so does the plain-data mirror the
// main thread keeps once training moves into a Web Worker (see
// lib/engine.ts), letting network-diagram.tsx/decision-boundary.tsx stay
// agnostic to which one they were handed.
export interface NetworkSnapshot {
  sizes: number[]
  layers: LayerLike[]
  lastActivations: Matrix[] | null
  predict(X: Matrix): Matrix
}

export interface ClassificationMetrics {
  confusionMatrix: number[][] // confusionMatrix[actual][predicted]
  precision: number[] // per class, TP / (TP + FP)
  recall: number[] // per class, TP / (TP + FN)
  f1: number[] // per class, harmonic mean of precision and recall
  support: number[] // per class, count of actual samples
}

function classIndex(row: number[], numClasses: number): number {
  if (numClasses === 2) return row[0] >= 0.5 ? 1 : 0
  let best = 0
  for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k
  return best
}

// Confusion matrix + per-class precision/recall/F1, computed from whatever
// labeled set the caller hands it (the validation split when one exists,
// the training set otherwise). Predicted class comes from a real forward
// pass through `network`, not from any cached activation.
export function computeMetrics(network: NetworkSnapshot, X: Matrix, Y: Matrix, numClasses: number): ClassificationMetrics {
  const predictions = network.predict(X)
  const confusionMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0))
  for (let i = 0; i < X.length; i++) {
    const actual = classIndex(Y[i], numClasses)
    const predicted = classIndex(predictions[i], numClasses)
    confusionMatrix[actual][predicted]++
  }

  const precision: number[] = []
  const recall: number[] = []
  const f1: number[] = []
  const support: number[] = []
  for (let c = 0; c < numClasses; c++) {
    let tp = confusionMatrix[c][c]
    let fp = 0
    let fn = 0
    let actualCount = 0
    for (let k = 0; k < numClasses; k++) {
      actualCount += confusionMatrix[c][k]
      if (k !== c) {
        fn += confusionMatrix[c][k]
        fp += confusionMatrix[k][c]
      }
    }
    const p = tp + fp === 0 ? 0 : tp / (tp + fp)
    const r = tp + fn === 0 ? 0 : tp / (tp + fn)
    precision.push(p)
    recall.push(r)
    f1.push(p + r === 0 ? 0 : (2 * p * r) / (p + r))
    support.push(actualCount)
  }

  return { confusionMatrix, precision, recall, f1, support }
}

const EPS = 1e-7 // clamps log() away from the domain edges in cross-entropy

// Shared forward pass, used by Network.forward and by the main-thread
// snapshot's predict() once the live Network only exists inside a worker.
export function forwardPass(layers: LayerLike[], X: Matrix): Matrix[] {
  const activations: Matrix[] = [X]
  let current = X
  for (const layer of layers) {
    const z = addBias(multiply(current, layer.weights), layer.biases)
    current = layer.activation === "softmax" ? softmaxRows(z) : map(z, ACTIVATIONS[layer.activation].fn)
    activations.push(current)
  }
  return activations
}

export class Network implements NetworkSnapshot {
  layers: Layer[] = []
  readonly sizes: number[]
  readonly hiddenActivation: Activation
  // Per-layer activations from the most recent trainStep, read directly by
  // the network-diagram visualization so it doesn't need its own forward
  // pass just to color neurons.
  lastActivations: Matrix[] | null = null

  constructor(sizes: number[], hiddenActivation: Activation = "tanh") {
    this.sizes = sizes
    this.hiddenActivation = hiddenActivation
    // `sizes.at(-1)` is the actual output layer width, which the caller
    // sets to 1 for binary (a single sigmoid unit reads directly as
    // P(class=1), the original design) or to numClasses for 3+ classes
    // (a full softmax layer). Only the width itself signals which mode
    // this is, there is no separate "how many classes" input here.
    const outputSize = sizes.at(-1)!
    for (let i = 0; i < sizes.length - 1; i++) {
      const fanIn = sizes[i]
      const fanOut = sizes[i + 1]
      const isOutputLayer = i === sizes.length - 2
      this.layers.push({
        weights: randomMatrix(fanIn, fanOut, fanIn),
        biases: new Array(fanOut).fill(0),
        activation: isOutputLayer ? (outputSize === 1 ? "sigmoid" : "softmax") : hiddenActivation,
      })
    }
  }

  // 2 for a binary (1-unit sigmoid) output layer, otherwise the softmax
  // layer's own width. This is "how many classes", not "the output
  // layer's width" (those only differ in the binary case).
  get numClasses(): number {
    const outputSize = this.sizes.at(-1)!
    return outputSize === 1 ? 2 : outputSize
  }

  forward(X: Matrix): ForwardCache {
    return { activations: forwardPass(this.layers, X) }
  }

  predict(X: Matrix): Matrix {
    return this.forward(X).activations.at(-1)!
  }

  // Forward pass + loss, no weight update. Used for validation loss (the
  // overfitting detector) and for the before/after readout around pruning,
  // neither of which should touch the weights the way trainStep does.
  evaluate(X: Matrix, Y: Matrix): number {
    return this.loss(this.predict(X), Y)
  }

  // Zeroes the smallest-magnitude weights across every layer, a global
  // percentile cutoff rather than a per-layer one, so a layer that's
  // already sparse isn't forced to lose more just to hit the same fraction
  // as a dense one.
  pruneWeakWeights(fraction: number) {
    const magnitudes: number[] = []
    for (const layer of this.layers) for (const row of layer.weights) for (const w of row) magnitudes.push(Math.abs(w))
    magnitudes.sort((a, b) => a - b)
    const cutoffIndex = Math.floor(magnitudes.length * fraction)
    const threshold = magnitudes[cutoffIndex] ?? 0
    for (const layer of this.layers) {
      layer.weights = layer.weights.map((row) => row.map((w) => (Math.abs(w) < threshold ? 0 : w)))
    }
  }

  // Binary cross-entropy for the 2-class case (unchanged from the original
  // design), categorical cross-entropy for 3+ classes with one-hot targets.
  private loss(predicted: Matrix, target: Matrix): number {
    let total = 0
    const n = predicted.length
    if (this.numClasses === 2) {
      for (let i = 0; i < n; i++) {
        const p = Math.min(1 - EPS, Math.max(EPS, predicted[i][0]))
        const y = target[i][0]
        total += -(y * Math.log(p) + (1 - y) * Math.log(1 - p))
      }
    } else {
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < this.numClasses; k++) {
          const p = Math.min(1 - EPS, Math.max(EPS, predicted[i][k]))
          total += -target[i][k] * Math.log(p)
        }
      }
    }
    return total / n
  }

  // One SGD step over the given batch. Mutates the network's weights in
  // place and returns the pre-update loss + the forward cache (consumed by
  // the visualization layer to read per-neuron activations this tick).
  trainStep(X: Matrix, Y: Matrix, learningRate: number): TrainResult {
    const cache = this.forward(X)
    const predicted = cache.activations.at(-1)!
    const loss = this.loss(predicted, Y)
    const batchSize = X.length

    // dL/dz for the output layer collapses to (predicted - target) for
    // both pairings this network supports: sigmoid + binary cross-entropy,
    // and softmax + categorical cross-entropy. The two loss derivatives
    // cancel algebraically against their matching activation's derivative
    // in both cases, which is *why* each pairing is the standard choice
    // rather than an arbitrary one, not a coincidence that lets one delta
    // formula cover both output modes below.
    let delta: Matrix = elementwise(predicted, Y, (p, y) => p - y)

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l]
      const prevActivation = cache.activations[l] // this layer's input
      const gradWeights = scale(multiply(transpose(prevActivation), delta), 1 / batchSize)
      const gradBiases = sumColumns(delta).map((s) => s / batchSize)

      // Propagate delta to the previous layer before this layer's weights
      // are updated (the update must use the gradient computed from the
      // *old* weights, not weights already mutated this step).
      if (l > 0) {
        const dPrevActivation = multiply(delta, transpose(layer.weights))
        const prevLayerActivated = cache.activations[l] // == this layer's input
        const prevActivationType = this.layers[l - 1].activation
        if (prevActivationType === "softmax") {
          // Loop invariant: softmax only ever sits on the output layer, so
          // layers[l - 1] (always a hidden layer, l > 0 here) can never be
          // it. This throw documents that invariant rather than silently
          // trusting it.
          throw new Error("softmax is only valid as the output layer's activation")
        }
        const { derivative } = ACTIVATIONS[prevActivationType]
        delta = elementwise(dPrevActivation, prevLayerActivated, (d, a) => d * derivative(a))
      }

      layer.weights = elementwise(layer.weights, gradWeights, (w, g) => w - learningRate * g)
      layer.biases = layer.biases.map((b, j) => b - learningRate * gradBiases[j])
    }

    this.lastActivations = cache.activations
    return { loss, cache }
  }

  // Deep-enough snapshot for the visualization to read without risking a
  // torn read mid-update (weights/biases arrays are replaced, not mutated
  // in place, by trainStep above, so a shallow copy per layer is enough).
  snapshot() {
    return this.layers.map((l) => ({ weights: l.weights, biases: l.biases, activation: l.activation }))
  }
}
