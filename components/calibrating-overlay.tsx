"use client"

// Shown over a canvas panel between mount and its first real engine tick
// (the synchronous draw call before subscribing paints "nothing yet", not
// this) so INITIATE doesn't hand the operator a blank rectangle.
export function CalibratingOverlay() {
  return (
    <div className="absolute inset-0 mnemo-calibrating flex items-center justify-center pointer-events-none">
      <span className="mnemo-label animate-[mnemo-flicker_2s_linear_infinite]">Calibrating...</span>
    </div>
  )
}
