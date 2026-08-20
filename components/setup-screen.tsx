"use client"

import { useState } from "react"
import { DatasetPainter } from "@/components/dataset-painter"
import { Slider, Toggle } from "@/components/field-controls"
import { LrSchedulePreview } from "@/components/lr-schedule-preview"
import type { EngineConfig, StopCondition } from "@/lib/engine"
import type { LrSchedule } from "@/lib/lr-schedule"
import { type CustomPoint, DEFAULT_DATASET_NOISE, type DatasetId, MIN_POINTS_PER_CLASS } from "@/lib/nn/datasets"
import type { Activation } from "@/lib/nn/network"
import { ARCHITECTURE_PRESETS, PROTOCOL_PRESETS } from "@/lib/protocol"
import { cn } from "@/lib/utils"

interface SetupScreenProps {
  initialConfig: EngineConfig
  onInitiate: (config: EngineConfig) => void
}

const DATASETS: { id: DatasetId; label: string }[] = [
  { id: "xor", label: "XOR" },
  { id: "circles", label: "RINGS" },
  { id: "spirals", label: "SPIRAL" },
  { id: "triple", label: "TRIPLE" },
  { id: "custom", label: "CUSTOM" },
]

const ACTIVATIONS: { id: Activation; label: string }[] = [
  { id: "relu", label: "RELU" },
  { id: "tanh", label: "TANH" },
  { id: "sigmoid", label: "SIGM" },
]

const LOSS_TARGETS = [0.1, 0.05, 0.02, 0.01]
const STEP_TARGETS = [500, 2000, 5000, 20000]
const MAX_CUSTOM_LAYERS = 4

function StopModeButton({
  active,
  label,
  onClick,
}: {
  active: boolean
  label: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex-1 border px-2 py-1.5 text-[0.65rem] tracking-wider transition-colors cursor-pointer",
        active
          ? "border-accent bg-accent/15 text-accent"
          : "border-border text-muted-foreground hover:border-accent/50 hover:text-foreground",
      )}
    >
      {label}
    </button>
  )
}

export function SetupScreen({ initialConfig, onInitiate }: SetupScreenProps) {
  const [datasetId, setDatasetId] = useState(initialConfig.datasetId)
  const [activation, setActivation] = useState(initialConfig.activation)
  const [hiddenLayers, setHiddenLayers] = useState<number[]>(initialConfig.hiddenLayers)
  const [archMode, setArchMode] = useState<"guided" | "custom">("guided")
  const [learningRate, setLearningRate] = useState(initialConfig.learningRate)
  const [noise, setNoise] = useState(initialConfig.noise ?? DEFAULT_DATASET_NOISE)
  const [customPoints, setCustomPoints] = useState<CustomPoint[]>(initialConfig.customPoints ?? [])
  const [customNumClasses, setCustomNumClasses] = useState(initialConfig.customNumClasses ?? 2)

  const [lrScheduleType, setLrScheduleType] = useState<LrSchedule["type"]>(initialConfig.lrSchedule.type)
  const [stepEvery, setStepEvery] = useState(initialConfig.lrSchedule.type === "step" ? initialConfig.lrSchedule.every : 500)
  const [stepFactor, setStepFactor] = useState(initialConfig.lrSchedule.type === "step" ? initialConfig.lrSchedule.factor : 0.5)
  const [cosineTotalSteps, setCosineTotalSteps] = useState(
    initialConfig.lrSchedule.type === "cosine" ? initialConfig.lrSchedule.totalSteps : 2000,
  )
  const lrSchedule: LrSchedule =
    lrScheduleType === "step"
      ? { type: "step", every: stepEvery, factor: stepFactor }
      : lrScheduleType === "cosine"
        ? { type: "cosine", totalSteps: cosineTotalSteps }
        : { type: "none" }

  const initialStop = initialConfig.stopCondition
  const [stopMode, setStopMode] = useState<StopCondition["type"]>(initialStop.type)
  const [lossTarget, setLossTarget] = useState(initialStop.type === "loss" ? initialStop.target : 0.02)
  const [stepsTarget, setStepsTarget] = useState(initialStop.type === "steps" ? initialStop.target : 2000)

  const applyPreset = (presetId: string) => {
    const preset = PROTOCOL_PRESETS.find((p) => p.id === presetId)
    if (!preset) return
    setDatasetId(preset.config.datasetId)
    setActivation(preset.config.activation)
    setHiddenLayers(preset.config.hiddenLayers)
    setLearningRate(preset.config.learningRate)
    setLrScheduleType(preset.config.lrSchedule.type)
    setStopMode(preset.stopCondition.type)
    if (preset.stopCondition.type === "loss") setLossTarget(preset.stopCondition.target)
    if (preset.stopCondition.type === "steps") setStepsTarget(preset.stopCondition.target)
  }

  const setArchPreset = (archId: string) => {
    const preset = ARCHITECTURE_PRESETS.find((p) => p.id === archId)
    if (preset) setHiddenLayers(preset.layers)
  }

  const updateLayer = (index: number, size: number) => {
    setHiddenLayers((layers) => layers.map((l, i) => (i === index ? size : l)))
  }
  const addLayer = () => setHiddenLayers((layers) => (layers.length >= MAX_CUSTOM_LAYERS ? layers : [...layers, 8]))
  const removeLayer = (index: number) =>
    setHiddenLayers((layers) => (layers.length <= 1 ? layers : layers.filter((_, i) => i !== index)))

  const stopCondition: StopCondition =
    stopMode === "loss"
      ? { type: "loss", target: lossTarget }
      : stopMode === "steps"
        ? { type: "steps", target: stepsTarget }
        : { type: "manual" }

  // A custom specimen only unlocks INITIATE once every class the operator
  // selected actually has enough painted points to train on; every other
  // dataset is procedurally generated and always valid.
  const customValid =
    datasetId !== "custom" ||
    Array.from({ length: customNumClasses }, (_, k) => customPoints.filter((p) => p.cls === k).length).every(
      (count) => count >= MIN_POINTS_PER_CLASS,
    )

  const handleInitiate = () => {
    if (!customValid) return
    onInitiate({
      datasetId,
      hiddenLayers,
      activation,
      learningRate,
      stepsPerFrame: initialConfig.stepsPerFrame,
      noise: archMode === "custom" ? noise : undefined,
      customPoints: datasetId === "custom" ? customPoints : undefined,
      customNumClasses: datasetId === "custom" ? customNumClasses : undefined,
      lrSchedule,
      stopCondition,
    })
  }

  return (
    <div className="mnemo-panel p-4 sm:p-6 flex flex-col gap-5 max-w-3xl mx-auto w-full">
      <div>
        <div className="mnemo-header mb-2">Protocol Presets</div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {PROTOCOL_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => applyPreset(preset.id)}
              className="border border-border text-left px-3 py-2 hover:border-accent/50 hover:bg-accent/5 cursor-pointer transition-colors"
            >
              <div className="text-xs tracking-wider text-accent">{preset.label}</div>
              <div className="mnemo-label mt-1 normal-case tracking-normal text-[0.65rem] leading-snug">
                {preset.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        <div className="flex flex-col gap-1.5">
          <div className="mnemo-header">Specimen Set</div>
          <Toggle options={DATASETS} value={datasetId} onChange={setDatasetId} />
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="mnemo-header">Activation Function</div>
          <Toggle options={ACTIVATIONS} value={activation} onChange={setActivation} />
        </div>
      </div>

      {datasetId === "custom" && (
        <DatasetPainter
          points={customPoints}
          onPointsChange={setCustomPoints}
          numClasses={customNumClasses}
          onNumClassesChange={setCustomNumClasses}
        />
      )}

      <div className="flex flex-col gap-1.5">
        <div className="mnemo-header">
          <span>Architecture</span>
          <Toggle
            options={[
              { id: "guided" as const, label: "GUIDED" },
              { id: "custom" as const, label: "CUSTOM" },
            ]}
            value={archMode}
            onChange={setArchMode}
          />
        </div>

        {archMode === "guided" ? (
          <Toggle
            options={ARCHITECTURE_PRESETS.map((p) => ({ id: p.id, label: `${p.label} [${p.layers.join(",")}]` }))}
            value={ARCHITECTURE_PRESETS.find((p) => p.layers.join(",") === hiddenLayers.join(","))?.id ?? ""}
            onChange={setArchPreset}
          />
        ) : (
          <div className="flex flex-col gap-2 border border-border p-2.5">
            {hiddenLayers.map((size, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="mnemo-label w-10 shrink-0">L{i + 1}</span>
                <input
                  type="range"
                  className="mnemo-range flex-1"
                  min={2}
                  max={24}
                  step={1}
                  value={size}
                  onChange={(e) => updateLayer(i, Number(e.target.value))}
                />
                <span className="text-accent text-xs w-6 text-right">{size}</span>
                <button
                  onClick={() => removeLayer(i)}
                  disabled={hiddenLayers.length <= 1}
                  className="text-alert text-xs px-1.5 border border-alert/40 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer hover:bg-alert/10"
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              onClick={addLayer}
              disabled={hiddenLayers.length >= MAX_CUSTOM_LAYERS}
              className="mnemo-label border border-border py-1 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer hover:border-accent/50 hover:text-accent"
            >
              + ADD LAYER
            </button>
          </div>
        )}
      </div>

      <Slider
        label="Learning Rate"
        value={learningRate}
        display={learningRate.toFixed(2)}
        min={0.01}
        max={1}
        step={0.01}
        onChange={setLearningRate}
      />

      <div className="flex flex-col gap-1.5">
        <div className="mnemo-header">LR Schedule</div>
        <Toggle
          options={[
            { id: "none" as const, label: "NONE" },
            { id: "step" as const, label: "STEP DECAY" },
            { id: "cosine" as const, label: "COSINE" },
          ]}
          value={lrScheduleType}
          onChange={setLrScheduleType}
        />
        {lrScheduleType === "step" && (
          <div className="flex flex-col gap-2 border border-border p-2.5">
            <Slider
              label="Decay Every"
              value={stepEvery}
              display={`${stepEvery} steps`}
              min={100}
              max={2000}
              step={100}
              onChange={setStepEvery}
            />
            <Slider
              label="Decay Factor"
              value={stepFactor}
              display={stepFactor.toFixed(2)}
              min={0.1}
              max={0.9}
              step={0.05}
              onChange={setStepFactor}
            />
          </div>
        )}
        {lrScheduleType === "cosine" && (
          <div className="flex flex-col gap-2 border border-border p-2.5">
            <Slider
              label="Anneal Over"
              value={cosineTotalSteps}
              display={`${cosineTotalSteps} steps`}
              min={200}
              max={5000}
              step={100}
              onChange={setCosineTotalSteps}
            />
          </div>
        )}
        {lrScheduleType !== "none" && <LrSchedulePreview schedule={lrSchedule} baseLR={learningRate} />}
      </div>

      {archMode === "custom" && (
        <Slider
          label="Specimen Noise"
          value={noise}
          display={noise.toFixed(2)}
          min={0.02}
          max={0.25}
          step={0.01}
          onChange={setNoise}
        />
      )}

      <div className="flex flex-col gap-1.5">
        <div className="mnemo-header">Halt Condition</div>
        <div className="flex gap-1.5">
          <StopModeButton active={stopMode === "loss"} label="LOSS TARGET" onClick={() => setStopMode("loss")} />
          <StopModeButton active={stopMode === "steps"} label="MAX STEPS" onClick={() => setStopMode("steps")} />
          <StopModeButton active={stopMode === "manual"} label="MANUAL" onClick={() => setStopMode("manual")} />
        </div>
        {stopMode === "loss" && (
          <div className="flex gap-1.5 mt-1">
            {LOSS_TARGETS.map((t) => (
              <StopModeButton key={t} active={lossTarget === t} label={`<${t}`} onClick={() => setLossTarget(t)} />
            ))}
          </div>
        )}
        {stopMode === "steps" && (
          <div className="flex gap-1.5 mt-1">
            {STEP_TARGETS.map((t) => (
              <StopModeButton key={t} active={stepsTarget === t} label={String(t)} onClick={() => setStepsTarget(t)} />
            ))}
          </div>
        )}
        {stopMode === "manual" && (
          <p className="mnemo-label normal-case tracking-normal text-[0.65rem] leading-snug mt-1">
            Runs indefinitely. Pause or abort from the live controls once the sequence starts.
          </p>
        )}
      </div>

      <button
        onClick={handleInitiate}
        disabled={!customValid}
        className="border border-accent bg-accent/15 text-accent px-4 py-2.5 text-sm tracking-[0.2em] cursor-pointer hover:bg-accent/25 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-accent/15"
      >
        INITIATE SEQUENCE
      </button>
    </div>
  )
}
