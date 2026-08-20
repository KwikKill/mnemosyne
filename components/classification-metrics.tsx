"use client"

import type { ClassificationMetrics } from "@/lib/nn/network"

interface ClassificationMetricsProps {
  metrics: ClassificationMetrics
}

export function ClassificationMetricsTable({ metrics }: ClassificationMetricsProps) {
  const numClasses = metrics.precision.length

  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3">
      <div className="mnemo-label">Classification Metrics</div>

      <div className="overflow-x-auto">
        <table className="w-full text-[0.65rem] border-collapse">
          <thead>
            <tr>
              <th className="text-left text-muted-foreground font-normal pr-2 pb-1">class</th>
              <th className="text-right text-muted-foreground font-normal px-1.5 pb-1">precision</th>
              <th className="text-right text-muted-foreground font-normal px-1.5 pb-1">recall</th>
              <th className="text-right text-muted-foreground font-normal px-1.5 pb-1">f1</th>
              <th className="text-right text-muted-foreground font-normal pl-1.5 pb-1">n</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: numClasses }, (_, c) => (
              <tr key={c} className="border-t border-border/60">
                <td className="text-foreground py-1 pr-2">{c}</td>
                <td className="text-accent text-right px-1.5 py-1">{metrics.precision[c].toFixed(2)}</td>
                <td className="text-accent text-right px-1.5 py-1">{metrics.recall[c].toFixed(2)}</td>
                <td className="text-accent text-right px-1.5 py-1">{metrics.f1[c].toFixed(2)}</td>
                <td className="text-muted-foreground text-right pl-1.5 py-1">{metrics.support[c]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mnemo-label normal-case tracking-normal text-[0.6rem]">confusion matrix (actual / predicted)</div>
      <div className="overflow-x-auto">
        <table className="border-collapse text-[0.6rem]">
          <tbody>
            {metrics.confusionMatrix.map((row, actual) => (
              <tr key={actual}>
                {row.map((count, predicted) => (
                  <td
                    key={predicted}
                    className="w-7 h-6 text-center border border-border/60"
                    style={{
                      color: actual === predicted ? "var(--accent)" : "var(--muted-foreground)",
                      backgroundColor:
                        actual === predicted
                          ? `color-mix(in oklch, var(--accent) ${Math.min(60, count * 6)}%, transparent)`
                          : undefined,
                    }}
                  >
                    {count}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
