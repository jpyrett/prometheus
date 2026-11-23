import { useEffect, useRef, useState } from "react";
import type { RunRequest, TradeMode } from "../api/types";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";
import { Check } from "./ui/Check";
import { Switch } from "./ui/Switch";
import { SmallGear } from "./Gears";

export interface RequestCardProps {
  prompt: string;
  onPrompt: (text: string) => void;
  maxChars: number;
  /** The account to run on, or null (Go! is disabled until there is one). */
  accountId: number | null;
  running: boolean;
  starting: boolean;
  defaults: { strict: boolean; neutral: boolean; fetch: boolean; raise_funds?: boolean; ignore_critic?: boolean };
  onGo: (request: RunRequest) => void;
  onStop: () => void;
  error: string | null;
  /** Why Go! cannot be pressed even with text and an account (a sample account is read-only), shown as its tooltip. */
  blockedReason?: string | null;
  /** The trade mode on show: the account's (the account is the truth), or the page's own value while there is no account yet. */
  mode: TradeMode;
  /** Save the mode (the account's `PATCH {mode}`, which reaches the Account Manager); a refusal rejects and the switch moves back. */
  onMode: (mode: TradeMode) => Promise<unknown> | void;
  /** Why the Mode switch cannot be used (a sample account, a recorded session): shown as its tooltip. A run that is active disables it too. */
  modeLockedReason?: string | null;
  /** A recorded session: nothing is fetched, so the Refresh box is shown unticked and disabled. */
  replay?: boolean;
}

const MODE_TIPS: Record<TradeMode, string> = { ask: "Ask: nothing trades without your yes.", auto: "Auto: the desk trades what the Critic approves without asking." };

/** The request in the user's own words (spec 5.3 (2) (6) (7) (8)). */
export function RequestCard({ prompt, onPrompt, maxChars, accountId, running, starting, defaults, onGo, onStop, error, blockedReason = null, mode, onMode, modeLockedReason = null, replay = false }: RequestCardProps) {
  const [strict, setStrict] = useState(defaults.strict);
  const [neutral, setNeutral] = useState(defaults.neutral);
  const [fetch, setFetch] = useState(defaults.fetch);
  const [raiseFunds, setRaiseFunds] = useState(defaults.raise_funds ?? false);      // "Always raise funds" (entry 142, R12): off unless ticked
  const [ignoreCritic, setIgnoreCritic] = useState(defaults.ignore_critic ?? false);      // "Ignore Critic" (entry 148): skip the Critic's step for this run; off unless ticked
  const refresh = replay ? false : fetch;                  // a recorded session fetches nothing
  // The switch moves first; if the save is refused it moves back and the server's message shows under it.
  const [shown, setShown] = useState<TradeMode>(mode);
  const [modeError, setModeError] = useState<string | null>(null);
  useEffect(() => setShown(mode), [mode]);
  const box = useRef<HTMLTextAreaElement>(null);
  const busy = running || starting;
  const canGo = prompt.trim().length > 0 && accountId !== null && !busy && !blockedReason;
  // Strict means only what the Critic approves trades: neutral passes are off whatever the other box says. And the Critic does not run with Ignore Critic, so Strict cannot be on with it (IC9): it shows unticked and disabled.
  const effectiveStrict = strict && !ignoreCritic;
  const effectiveNeutral = effectiveStrict ? false : neutral;

  const go = () => {
    if (!canGo || accountId === null) return;
    onGo({ account_id: accountId, prompt, strict: effectiveStrict, neutral: effectiveNeutral, fetch: refresh, raise_funds: raiseFunds, ignore_critic: ignoreCritic, mode: shown });
  };

  async function flip(on: boolean) {
    const next: TradeMode = on ? "auto" : "ask";
    const before = mode;
    setModeError(null);
    setShown(next);
    try {
      await onMode(next);
    } catch (e) {
      setShown(before);
      setModeError(e instanceof Error && e.message ? e.message : "the mode could not be saved");
    }
  }

  return (
    <Card title="Your request" right="in your own words">
      <div className="card-b">
        <div style={{ position: "relative" }}>
          <textarea
            ref={box}
            className="in"
            maxLength={maxChars}
            aria-label="Your request"
            value={prompt}
            onChange={(e) => onPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                go();
              }
            }}
          />
          <span className="count" aria-hidden="true">
            {prompt.length} / {maxChars}
          </span>
        </div>
        <div className="reqbar">
          <div className="checks">
            <Check
              label="Strict recommendations"
              checked={effectiveStrict}
              disabled={busy || ignoreCritic}
              onChange={(e) => setStrict(e.target.checked)}
              title={ignoreCritic ? "Ignore Critic: the Critic does not run" : "Only recommendations the Critic approves may trade. A neutral recommendation never passes."}
            />
            <Check
              label="Neutral recommendations as passes"
              checked={effectiveNeutral}
              disabled={busy || effectiveStrict}
              onChange={(e) => setNeutral(e.target.checked)}
              title={effectiveStrict ? "Strict: only what the Critic approves trades" : "A company you named that the Advisor scored neutral counts as approved (ALLOW_NEUTRAL_RECOMMENDATIONS, for this run)"}
            />
            <Check
              label="Ignore Critic"
              checked={ignoreCritic}
              disabled={busy}
              onChange={(e) => setIgnoreCritic(e.target.checked)}
              title="Skip the Critic for this run: it does not review anything, and every recommendation the Advisor makes counts as approved, including one the Critic would have denied. Auto mode trades them; Ask mode asks you about each and says the Critic did not review it."
            />
            <Check
              label="Always raise funds"
              checked={raiseFunds}
              disabled={busy}
              onChange={(e) => setRaiseFunds(e.target.checked)}
              title={'The same as saying "raise funds if needed" in every request (ALWAYS_RAISE_FUNDS, for this run): when the cash does not cover a purchase you typed, the Account Manager sells holdings to pay for it. Ask mode still shows you the plan and asks you to approve each trade; Auto mode sells and buys. "Don\'t sell anything" in your words still wins.'}
            />
            <Check
              label={
                <>
                  Refresh market data first
                </>
              }
              checked={refresh}
              disabled={busy || replay}
              onChange={(e) => setFetch(e.target.checked)}
              title={replay ? "a recorded session: nothing is fetched" : "Fetches fresh bars, news and filings with the paid APIs and writes the data files. Untick to use the stored data."}
            />
          </div>
          <div className="right">
            <div className="mode" title={busy ? "a run is active: wait for it to finish" : (modeLockedReason ?? MODE_TIPS[shown])}>
              <span className="k" id="mode-label">
                Mode
              </span>
              <span className={`opt${shown === "ask" ? " on" : ""}`}>Ask</span>
              <Switch checked={shown === "auto"} onChange={(on) => void flip(on)} disabled={busy || modeLockedReason !== null} label={`Trade mode: ${shown === "auto" ? "Auto" : "Ask"}`} />
              <span className={`opt${shown === "auto" ? " on" : ""}`}>Auto</span>
            </div>
            {modeError && (
              <p className="err mode-err" role="alert">
                {modeError}
              </p>
            )}
            <div className="acts">
              {busy ? (
                <Button variant="ghost" className="stop" onClick={onStop} disabled={starting && !running}>
                  Stop
                </Button>
              ) : (
                <Button
                  variant="ghost"
                  onClick={() => {
                    onPrompt("");
                    box.current?.focus();
                  }}
                >
                  Clear
                </Button>
              )}
              <Button variant="go" disabled={!canGo} onClick={go} title={blockedReason ?? (accountId === null ? "Open an account first" : "Ctrl+Enter")}>
                {busy ? (
                  <>
                    <SmallGear />
                    Running…
                  </>
                ) : (
                  "Go!"
                )}
              </Button>
            </div>
          </div>
        </div>
        {error && (
          <p className="err" role="alert">
            {error}
          </p>
        )}
      </div>
    </Card>
  );
}
