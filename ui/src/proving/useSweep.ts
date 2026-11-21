// The playhead's pace (spec 13.2): the points of the curve arrive as fast as the engine makes them and are DRAWN at the page's own speed, so a person can watch the test being walked.
// "Skip to the end" draws them all at once. A saved run that is simply opened is drawn whole (`animate` false), and so is everything under `prefers-reduced-motion`.
import { useCallback, useEffect, useRef, useState } from "react";
import { prefersReducedMotion } from "../charts/useSize";

export interface Sweep {
  /** How many of the `total` points are drawn. */
  shown: number;
  /** The playhead is moving (points are being drawn). */
  sweeping: boolean;
  skip: () => void;
}

export function useSweep(total: number, animate: boolean, tickMs = 45): Sweep {
  const still = !animate || prefersReducedMotion();
  const [shown, setShown] = useState(still ? total : 0);
  const skipped = useRef(false);

  useEffect(() => {
    if (total === 0) {
      setShown(0);
      skipped.current = false;
    }
  }, [total]);

  useEffect(() => {
    if (still || skipped.current) {
      setShown(total);
      return;
    }
    if (shown >= total) return;
    const id = setInterval(() => {
      setShown((s) => {
        // a long backlog is caught up in a few seconds: the pace speeds up with what is waiting
        const step = Math.max(1, Math.ceil((total - s) / 140));
        return Math.min(total, s + step);
      });
    }, tickMs);
    return () => clearInterval(id);
  }, [still, total, shown, tickMs]);

  const skip = useCallback(() => {
    skipped.current = true;
    setShown(total);
  }, [total]);

  const drawn = still || skipped.current ? total : Math.min(shown, total);
  return { shown: drawn, sweeping: !still && drawn < total, skip };
}
