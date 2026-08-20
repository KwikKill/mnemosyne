// Quick-fill presets for the setup screen. None of these skip
// configuration, they just seed the fields with a coherent combination.
import type { EngineConfig, StopCondition } from "./engine"

// Raw RGB for up to 4 classes, shared by decision-boundary.tsx's heatmap
// and dataset-painter.tsx's point picker so a class always reads as the
// same color everywhere it appears. Index 0/1 match the original binary
// pair (cream/red); 2/3 extend into the theme's hot-orange and amber
// tones rather than reaching for an unrelated hue, staying "in red tones"
// even at 4 classes.
export const CLASS_PALETTE: [number, number, number][] = [
  [225, 208, 195], // class 0: cream (foreground)
  [198, 54, 48], // class 1: red (accent)
  [232, 112, 63], // class 2: hot orange (alert)
  [196, 156, 74], // class 3: amber
]

export interface ArchitecturePreset {
  id: string
  label: string
  layers: number[]
}

export const ARCHITECTURE_PRESETS: ArchitecturePreset[] = [
  { id: "shallow", label: "SHALLOW", layers: [8] },
  { id: "standard", label: "STANDARD", layers: [8, 8] },
  { id: "deep", label: "DEEP", layers: [12, 12, 6] },
]

export interface ProtocolPreset {
  id: string
  label: string
  description: string
  config: Omit<EngineConfig, "stopCondition">
  stopCondition: StopCondition
}

export const PROTOCOL_PRESETS: ProtocolPreset[] = [
  {
    id: "standard",
    label: "STANDARD SEQUENCE",
    description: "Concentric rings, standard architecture, reliable convergence.",
    config: {
      datasetId: "circles",
      hiddenLayers: [8, 8],
      learningRate: 0.25,
      activation: "tanh",
      stepsPerFrame: 6,
      lrSchedule: { type: "none" },
    },
    stopCondition: { type: "loss", target: 0.02 },
  },
  {
    id: "stress",
    label: "STRESS TEST",
    description: "Twin spiral: a harder boundary the network may never fully resolve.",
    config: {
      datasetId: "spirals",
      hiddenLayers: [8, 8],
      learningRate: 0.25,
      activation: "tanh",
      stepsPerFrame: 6,
      lrSchedule: { type: "none" },
    },
    stopCondition: { type: "steps", target: 5000 },
  },
  {
    id: "minimal",
    label: "MINIMAL CAPACITY",
    description: "XOR through a single 2-unit hidden layer, likely underfitting by design.",
    config: {
      datasetId: "xor",
      hiddenLayers: [2],
      learningRate: 0.25,
      activation: "tanh",
      stepsPerFrame: 6,
      lrSchedule: { type: "none" },
    },
    stopCondition: { type: "steps", target: 2000 },
  },
  {
    id: "trichotomy",
    label: "TRICHOTOMY",
    description: "Three clusters, softmax output: the multi-class path instead of binary sigmoid.",
    config: {
      datasetId: "triple",
      hiddenLayers: [8, 8],
      learningRate: 0.25,
      activation: "tanh",
      stepsPerFrame: 6,
      lrSchedule: { type: "none" },
    },
    stopCondition: { type: "steps", target: 3000 },
  },
]

export const DEFAULT_PROTOCOL = PROTOCOL_PRESETS[0]
