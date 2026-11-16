// Inline SVG icons (spec 5.1): no icon font, no network.
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = { fill: "none", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true } as const;

export function CheckIcon({ size = 16, ...p }: P & { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth={3} {...base} {...p}>
      <path d="m5 12 5 5 9-10" />
    </svg>
  );
}

export function CrossIcon({ size = 16, ...p }: P & { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth={3} {...base} {...p}>
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function SunIcon({ size = 16, ...p }: P & { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth={2} {...base} {...p}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" />
    </svg>
  );
}

export function MoonIcon({ size = 16, ...p }: P & { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth={2} {...base} {...p}>
      <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" />
    </svg>
  );
}

export function ChevronIcon({ size = 14, ...p }: P & { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth={2.5} {...base} {...p}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function Logo() {
  return (
    <div className="logo" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 2c1 4-3 5.5-3 9.5 0 1.3.6 2.3 1.4 3-.2-2.2 1-3.6 2.1-4.7.3 2.6 2.5 3.4 2.5 6.2 0 .8-.2 1.5-.5 2.1 2-1 3.5-3.1 3.5-5.9C18 7.6 13.5 6.5 12 2Z" fill="#1B0E05" />
        <path d="M8 13.2c-1.2 1.2-2 2.7-2 4.5C6 20.6 8.7 22 12 22s6-1.4 6-4.3" stroke="#1B0E05" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    </div>
  );
}
