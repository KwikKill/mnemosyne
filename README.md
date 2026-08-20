# Mnemosyne

A hand-written multilayer perceptron, trained live in the browser and
inspected through a Signalis-inspired station-terminal interface. No
inference API, no ML library: forward pass, backpropagation, and gradient
descent are all written out by hand in `lib/nn/`, so the math is
inspectable, not just callable.

Live at `mnemosyne.somi.blaisot.org` in production, `mnemosyne.somi.localhost`
in the beta stack.

## What it does

Configure a training run (dataset, architecture, activation, learning rate,
stop condition) in a setup screen, watch it train against a live network
diagram, decision-boundary heatmap, and loss chart, then inspect the
trained network once it halts: click anywhere on the classification
surface to probe it, prune its weakest synapses and see the loss impact,
scrub back through its training history, or download a clip of its
convergence.

## Architecture

**Training runs in a Web Worker, not the main thread.**
`lib/training-worker.ts` owns the real `Network` and `Dataset` instances
and drives the SGD loop, paced with `setTimeout` (dedicated workers don't
reliably get `requestAnimationFrame`, and it decouples training speed from
the page's render cadence). `lib/engine.ts`'s `TrainingEngine` is a
main-thread shim: it only ever sends commands (`play`, `pause`, `reset`,
`prune`, ...) and receives serialized ticks back over `postMessage`. Every
UI component talks to `TrainingEngine`, never to the worker directly, and
never to a live `Network` instance, since the real one lives on the other
side of the thread boundary. `lib/worker-protocol.ts` defines the shared
message shapes both sides import, so a change on one side of `postMessage`
shows up as a type error on the other.

**The network math (`lib/nn/`)**:
- `matrix.ts`: dependency-free `number[][]` matrix/vector operations.
- `network.ts`: the `Network` class, forward pass, backward pass, SGD
  update, evaluation, pruning. Binary classification uses a single sigmoid
  output unit with binary cross-entropy (the original design); three or
  more classes get a full softmax output layer with categorical
  cross-entropy instead. The two modes share almost the entire backward
  pass: `predicted - target` is the output-layer gradient shortcut for
  *both* pairings, a textbook result (the loss derivative and the matching
  activation's derivative cancel algebraically), not a coincidence that
  happens to let one line of code cover both cases.
- `datasets.ts`: procedural toy datasets (XOR, concentric rings, twin
  spiral, three clusters) plus the custom dataset painter's point-packaging
  function, and the train/validation split used for the overfitting
  detector.

**Visualization components** (`components/network-diagram.tsx`,
`decision-boundary.tsx`, `loss-chart.tsx`, `weight-histogram.tsx`) each
subscribe to `TrainingEngine` and draw straight to a `<canvas>` ref on every
tick, bypassing React state for anything that changes every frame. They can
also render a historical point from the replay scrubber instead of the live
network (an `overrideNetwork` prop), which is how "watch it train, then
scrub back through what happened" works without a second rendering path.

## Running locally

```bash
pnpm install
pnpm dev
```

Or via this repo's Docker workflow (see the root `docker-compose*.yml` and
`docker-compose-beta.yml`): this project follows the same standalone-build
pattern as the other sites in this repo, no database, no Prisma.

## Tests

```bash
pnpm test
```

Runs Vitest over `lib/nn/matrix.ts` and `lib/nn/network.ts`: matrix
operation correctness, forward-pass output shapes for both the binary and
softmax paths, a loss-decreases-over-training sanity check on a learnable
toy problem, and a finite-difference gradient check that compares
`trainStep`'s analytical gradient against a numerical approximation of the
same loss surface: the strongest evidence the hand-written backward pass is
actually correct, not just "looks like it converges" in the UI.
