// Toy 2D classification datasets, generated client-side. Points live in
// [-1, 1] x [-1, 1]. Two-class datasets label rows with a scalar 0/1 (the
// network's binary sigmoid path reads that directly); three-or-more-class
// datasets one-hot encode instead (the softmax path's categorical
// cross-entropy expects that shape).
import type { Matrix } from "./matrix"

export interface Dataset {
  id: DatasetId
  label: string
  X: Matrix
  Y: Matrix
  numClasses: number
}

export type DatasetId = "xor" | "circles" | "spirals" | "triple" | "custom"
// Procedural generation only covers the built-in patterns; "custom" comes
// from the dataset painter via buildCustomDataset instead, which callers
// must branch to explicitly (see lib/training-worker.ts).
export type ProceduralDatasetId = Exclude<DatasetId, "custom">

const N_PER_CLASS = 120
const DEFAULT_NOISE = 0.08

function jitter(v: number, amount: number) {
  return v + (Math.random() * 2 - 1) * amount
}

function oneHot(cls: number, numClasses: number): number[] {
  return Array.from({ length: numClasses }, (_, k) => (k === cls ? 1 : 0))
}

function makeXor(scale: number): Dataset {
  const X: number[][] = []
  const Y: number[][] = []
  for (let i = 0; i < N_PER_CLASS * 2; i++) {
    const quadrant = i % 4
    const baseX = quadrant === 0 || quadrant === 3 ? -0.5 : 0.5
    const baseY = quadrant === 0 || quadrant === 1 ? -0.5 : 0.5
    const label = quadrant === 0 || quadrant === 2 ? 0 : 1
    X.push([jitter(baseX, 0.3 * scale), jitter(baseY, 0.3 * scale)])
    Y.push([label])
  }
  return { id: "xor", label: "XOR", X, Y, numClasses: 2 }
}

function makeCircles(scale: number): Dataset {
  const X: number[][] = []
  const Y: number[][] = []
  for (let i = 0; i < N_PER_CLASS; i++) {
    const angle = Math.random() * Math.PI * 2
    const r = jitter(0.25, 0.08 * scale)
    X.push([Math.cos(angle) * r, Math.sin(angle) * r])
    Y.push([0])
  }
  for (let i = 0; i < N_PER_CLASS; i++) {
    const angle = Math.random() * Math.PI * 2
    const r = jitter(0.75, 0.08 * scale)
    X.push([Math.cos(angle) * r, Math.sin(angle) * r])
    Y.push([1])
  }
  return { id: "circles", label: "CONCENTRIC RINGS", X, Y, numClasses: 2 }
}

function makeSpirals(scale: number): Dataset {
  const X: number[][] = []
  const Y: number[][] = []
  const turns = 1.8
  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < N_PER_CLASS; i++) {
      const t = i / N_PER_CLASS
      const angle = t * turns * Math.PI * 2 + cls * Math.PI
      const r = t * 0.9
      X.push([jitter(Math.cos(angle) * r, 0.04 * scale), jitter(Math.sin(angle) * r, 0.04 * scale)])
      Y.push([cls])
    }
  }
  return { id: "spirals", label: "TWIN SPIRAL", X, Y, numClasses: 2 }
}

// Three clusters at 120-degree intervals around the origin. The first
// built-in multi-class specimen, one-hot labels for the softmax path.
function makeTriple(scale: number): Dataset {
  const X: number[][] = []
  const Y: number[][] = []
  const numClasses = 3
  for (let cls = 0; cls < numClasses; cls++) {
    const centerAngle = (cls / numClasses) * Math.PI * 2
    const cx = Math.cos(centerAngle) * 0.55
    const cy = Math.sin(centerAngle) * 0.55
    for (let i = 0; i < N_PER_CLASS; i++) {
      X.push([jitter(cx, 0.22 * scale), jitter(cy, 0.22 * scale)])
      Y.push(oneHot(cls, numClasses))
    }
  }
  return { id: "triple", label: "TRIPLE CLUSTER", X, Y, numClasses }
}

// `noise` defaults to the original calibrated amount; callers (the setup
// screen's custom mode) may override it, scaled relative to that default
// rather than as an absolute jitter radius, since each dataset's baseline
// spread differs.
export function generateDataset(id: ProceduralDatasetId, noise: number = DEFAULT_NOISE): Dataset {
  const scale = noise / DEFAULT_NOISE
  switch (id) {
    case "xor":
      return makeXor(scale)
    case "circles":
      return makeCircles(scale)
    case "spirals":
      return makeSpirals(scale)
    case "triple":
      return makeTriple(scale)
  }
}

export interface CustomPoint {
  x: number
  y: number
  cls: number
}

// Packages the dataset painter's user-placed points directly, no
// procedural generation involved. Label shape follows the same rule as
// the built-in generators: scalar for 2 classes, one-hot for 3+.
export function buildCustomDataset(points: CustomPoint[], numClasses: number): Dataset {
  const X = points.map((p) => [p.x, p.y])
  const Y = points.map((p) => (numClasses === 2 ? [p.cls] : oneHot(p.cls, numClasses)))
  return { id: "custom", label: "CUSTOM SPECIMEN", X, Y, numClasses }
}

export const DATASET_IDS: ProceduralDatasetId[] = ["xor", "circles", "spirals", "triple"]
export const DEFAULT_DATASET_NOISE = DEFAULT_NOISE
export const MIN_POINTS_PER_CLASS = 3
// Below this many total points, an 80/20 split leaves too few validation
// samples to mean anything, so the overfitting detector skips entirely
// rather than split a handful of painted points.
export const MIN_POINTS_FOR_VALIDATION = 20

export interface DatasetSplit {
  train: { X: Matrix; Y: Matrix }
  val: { X: Matrix; Y: Matrix }
}

// Shuffles indices before splitting, every built-in generator emits
// class-grouped rows (all of class 0, then all of class 1, ...), an
// unshuffled split would hand the validation set nothing but the last
// class or two.
export function splitDataset(dataset: Dataset, valFraction: number): DatasetSplit {
  const n = dataset.X.length
  const indices = Array.from({ length: n }, (_, i) => i)
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  const valCount = Math.round(n * valFraction)
  const valIndices = new Set(indices.slice(0, valCount))
  const train: { X: Matrix; Y: Matrix } = { X: [], Y: [] }
  const val: { X: Matrix; Y: Matrix } = { X: [], Y: [] }
  for (let i = 0; i < n; i++) {
    const bucket = valIndices.has(i) ? val : train
    bucket.X.push(dataset.X[i])
    bucket.Y.push(dataset.Y[i])
  }
  return { train, val }
}
