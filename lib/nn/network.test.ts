import { describe, expect, it } from "vitest"
import { type Activation, Network } from "./network"

describe("Network - construction", () => {
  it("binary output is a single sigmoid unit", () => {
    const net = new Network([2, 4, 1], "tanh")
    expect(net.layers).toHaveLength(2)
    expect(net.layers[0].weights.length).toBe(2)
    expect(net.layers[0].weights[0].length).toBe(4)
    expect(net.layers[1].weights.length).toBe(4)
    expect(net.layers[1].weights[0].length).toBe(1)
    expect(net.layers[0].activation).toBe("tanh")
    expect(net.layers[1].activation).toBe("sigmoid")
    expect(net.numClasses).toBe(2)
  })

  it("3+ classes get a softmax output layer sized to numClasses", () => {
    const net = new Network([2, 5, 4], "relu")
    expect(net.layers[1].activation).toBe("softmax")
    expect(net.layers[1].weights[0].length).toBe(4)
    expect(net.numClasses).toBe(4)
  })
})

describe("Network - forward pass shapes", () => {
  it("binary predict returns one probability in [0,1] per sample", () => {
    const net = new Network([2, 4, 1])
    const out = net.predict([
      [0.1, 0.2],
      [0.3, -0.4],
      [0, 0],
    ])
    expect(out).toHaveLength(3)
    for (const row of out) {
      expect(row).toHaveLength(1)
      expect(row[0]).toBeGreaterThanOrEqual(0)
      expect(row[0]).toBeLessThanOrEqual(1)
    }
  })

  it("multi-class predict returns a softmax row (sums to 1) per sample", () => {
    const net = new Network([2, 4, 3])
    const out = net.predict([
      [0.1, 0.2],
      [0.3, -0.4],
    ])
    for (const row of out) {
      expect(row).toHaveLength(3)
      const sum = row.reduce((a, b) => a + b, 0)
      expect(sum).toBeCloseTo(1, 8)
    }
  })
})

describe("Network - trainStep reduces loss on a learnable toy problem", () => {
  it("loss decreases over training on a linearly separable set", () => {
    const net = new Network([2, 4, 1], "tanh")
    const X: number[][] = []
    const Y: number[][] = []
    for (let i = 0; i < 40; i++) {
      const x0 = Math.random() * 2 - 1
      const x1 = Math.random() * 2 - 1
      X.push([x0, x1])
      Y.push([x0 > 0 ? 1 : 0])
    }
    const first = net.trainStep(X, Y, 0.5).loss
    let last = first
    for (let step = 0; step < 200; step++) {
      last = net.trainStep(X, Y, 0.5).loss
    }
    expect(last).toBeLessThan(first)
    expect(last).toBeLessThan(0.3)
  })
})

describe("Network - backprop gradient matches a numerical approximation", () => {
  // The strongest evidence the hand-written backward pass is actually
  // correct, not just "looks like it converges": compare the analytical
  // gradient trainStep applies against a finite-difference approximation
  // of the same loss surface, for both output modes (they take genuinely
  // different code paths through trainStep) and for a hidden-layer weight,
  // not just the output layer.
  function cloneWithWeights(sizes: number[], activation: Activation, weights: number[][][]): Network {
    const net = new Network(sizes, activation)
    net.layers.forEach((layer, i) => {
      layer.weights = weights[i].map((row) => [...row])
    })
    return net
  }

  function checkGradientAt(
    sizes: number[],
    activation: Activation,
    X: number[][],
    Y: number[][],
    layerIdx: number,
    i: number,
    j: number,
  ) {
    const net = new Network(sizes, activation)
    const originalWeights = net.layers.map((l) => l.weights.map((row) => [...row]))
    const before = originalWeights[layerIdx][i][j]

    // Analytical: trainStep applies `w - learningRate * gradient`, so the
    // gradient it used is recoverable from the weight change it made.
    const lr = 0.01
    net.trainStep(X, Y, lr)
    const after = net.layers[layerIdx].weights[i][j]
    const analytical = (before - after) / lr

    // Numerical: perturb only that one weight, holding every other weight
    // at its pre-trainStep value, and measure the loss surface directly.
    const eps = 1e-5
    const weightsPlus = originalWeights.map((l) => l.map((row) => [...row]))
    weightsPlus[layerIdx][i][j] = before + eps
    const lossPlus = cloneWithWeights(sizes, activation, weightsPlus).evaluate(X, Y)

    const weightsMinus = originalWeights.map((l) => l.map((row) => [...row]))
    weightsMinus[layerIdx][i][j] = before - eps
    const lossMinus = cloneWithWeights(sizes, activation, weightsMinus).evaluate(X, Y)

    const numerical = (lossPlus - lossMinus) / (2 * eps)
    return { analytical, numerical }
  }

  const X = [
    [0.5, -0.2],
    [0.1, 0.9],
    [-0.3, 0.4],
    [0.2, 0.2],
  ]

  it("matches for the binary sigmoid + cross-entropy output path", () => {
    const Y = [[1], [0], [1], [0]]
    const { analytical, numerical } = checkGradientAt([2, 3, 1], "tanh", X, Y, 1, 0, 0)
    expect(analytical).toBeCloseTo(numerical, 3)
  })

  it("matches for the softmax + categorical cross-entropy output path", () => {
    const Y = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ]
    const { analytical, numerical } = checkGradientAt([2, 3, 3], "tanh", X, Y, 1, 0, 0)
    expect(analytical).toBeCloseTo(numerical, 3)
  })

  it("matches for a hidden-layer weight, not just the output layer", () => {
    const Y = [[1], [0], [1], [0]]
    const { analytical, numerical } = checkGradientAt([2, 3, 1], "relu", X, Y, 0, 1, 2)
    expect(analytical).toBeCloseTo(numerical, 2)
  })
})
