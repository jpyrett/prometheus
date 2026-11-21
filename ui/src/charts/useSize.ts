// The width of a chart's container, kept up to date with a ResizeObserver. A chart is drawn in a viewBox the size of its container, so one pixel on the screen is one unit of the drawing and a line stays crisp.
// jsdom has no layout (every width is 0) and no ResizeObserver: the fallback width is then what the tests and the first paint draw.
import { useLayoutEffect, useRef, useState, type RefObject } from "react";

export function useWidth<T extends HTMLElement = HTMLDivElement>(fallback = 760): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const w = Math.floor(el.getBoundingClientRect().width);
      if (w > 0) setWidth((old) => (old === w ? old : w));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** True when the person asked for no motion (the page is complete without any). */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
