import type { ReactNode } from "react";

export type BadgeTone = "pos" | "neg" | "warn" | "info" | "neutral" | "brand";

export function Badge({ tone = "neutral", num = false, title, children }: { tone?: BadgeTone; num?: boolean; title?: string; children: ReactNode }) {
  return (
    <span className={`badge ${tone}${num ? " num" : ""}`} title={title}>
      {children}
    </span>
  );
}
