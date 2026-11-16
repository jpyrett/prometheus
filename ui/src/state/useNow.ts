// A clock for the counters (the busy stage's seconds, the run timer). It ticks only while something counts.
// `ClockContext` lets a screenshot or a test freeze it ("?fixture=bob&at=30" feeds a recorded stream and a fixed `now`).
import { createContext, useContext, useEffect, useState } from "react";

export const ClockContext = createContext<number | null>(null);

/** Server-compatible epoch seconds (float), refreshed every 200 ms while `active`. */
export function useNow(active: boolean): number {
  const frozen = useContext(ClockContext);
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active || frozen !== null) return;
    setNow(Date.now() / 1000);
    const id = setInterval(() => setNow(Date.now() / 1000), 200);
    return () => clearInterval(id);
  }, [active, frozen]);
  return frozen ?? now;
}
