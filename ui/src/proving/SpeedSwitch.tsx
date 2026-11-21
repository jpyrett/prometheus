// The speed of a test (spec 7.6, speed.py): Fast (the default) or Faithful, a two-way segmented control. Each side says in one plain line what it costs in time, from ITS OWN plan answer (free), and what it changes:
// fast asks the Advisor's model with its reasoning off and the Critic's at a lower effort; faithful asks them as the live desk does. Every rule, score and fill is the same code in both.
import type { KeyboardEvent } from "react";
import { SPEEDS, planTimeWords } from "./timing";
import type { PlanAnswer, Speed } from "./types";

export interface SpeedSwitchProps {
  value: Speed;
  onChange: (speed: Speed) => void;
  /** The plan of each speed (the server's own answer for it), or null while it is being read. */
  plans: Record<Speed, PlanAnswer | null>;
  disabled?: boolean;
}

export function SpeedSwitch({ value, onChange, plans, disabled = false }: SpeedSwitchProps) {
  const move = (e: KeyboardEvent<HTMLButtonElement>) => {
    const at = SPEEDS.findIndex((s) => s.value === value);
    const to = e.key === "ArrowRight" || e.key === "ArrowDown" ? at + 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? at - 1 : null;
    if (to === null) return;
    e.preventDefault();
    const next = SPEEDS[(to + SPEEDS.length) % SPEEDS.length]!.value;
    onChange(next);
    (e.currentTarget.parentElement?.querySelector(`[data-speed="${next}"]`) as HTMLElement | null)?.focus();
  };
  return (
    <div className="speedsw" role="radiogroup" aria-label="Speed of the test">
      {SPEEDS.map((s) => {
        const on = s.value === value;
        const time = planTimeWords(plans[s.value]);
        return (
          <button key={s.value} type="button" role="radio" aria-checked={on} data-speed={s.value} tabIndex={on ? 0 : -1} disabled={disabled} className={`speedsw-opt${on ? " on" : ""}`} onClick={() => onChange(s.value)} onKeyDown={move}>
            <b>
              {s.label}
              {s.value === "fast" && <small>default</small>}
            </b>
            <span className="speedsw-line">
              {time ?? "reading the plan…"} · {s.what}
            </span>
          </button>
        );
      })}
    </div>
  );
}
