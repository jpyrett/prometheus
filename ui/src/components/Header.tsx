import { useEffect, useRef, useState } from "react";
import type { AccountView, Health } from "../api/types";
import { cap, initials } from "../format";
import type { Theme } from "../state/useTheme";
import { Logo, MoonIcon, SunIcon } from "./icons";

export interface HeaderProps {
  health: Health | null;
  account: AccountView | null;
  theme: Theme;
  onToggleTheme: () => void;
  /** Switch to an account number. */
  onSwitch: (id: number) => Promise<unknown>;
  onNewAccount: () => void;
  busy: boolean;
}

export function Header({ health, account, theme, onToggleTheme, onSwitch, onNewAccount, busy }: HeaderProps) {
  const replay = health?.mode === "replay";
  return (
    <>
      <header className="top">
        <Logo />
        <div className="brand">
          PROMETHEUS<small>Agentic trading desk</small>
        </div>
        <div className="spacer" />
        <span className="pill info" title="The Trader fills against a simulated broker. Nothing is sent to a real one.">
          PAPER TRADING · SIMULATED BROKER
        </span>
        {health?.clock_pill ? (
          <span className="pill warn" title={health.clock_note || undefined}>
            {health.clock_pill}
          </span>
        ) : null}
        {replay && (
          <span className="pill" title="a recorded session: no model is called">
            REPLAY
          </span>
        )}
        {health?.tracing === "langfuse" && (
          <span className="pill traced" title="This run is traced in Langfuse">
            <i className="dot" style={{ ["--agent" as string]: "var(--pos)" }} />
            Traced
          </span>
        )}
        <button type="button" className="icon-btn" onClick={onToggleTheme} title="Dark / light" aria-label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"}>
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
        <AccountChip account={account} replay={replay} onSwitch={onSwitch} onNewAccount={onNewAccount} busy={busy} />
      </header>
      {health && health.book !== "ok" && (
        <div className="bookbar" role="alert">
          <b>The account book is not available:</b> {health.book}. Run <code>scripts/setup_postgres.sh</code>.
        </div>
      )}
    </>
  );
}

function AccountChip({ account, replay, onSwitch, onNewAccount, busy }: { account: AccountView | null; replay: boolean; onSwitch: (id: number) => Promise<unknown>; onNewAccount: () => void; busy: boolean }) {
  const [open, setOpen] = useState(false);
  const [number, setNumber] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const a = account?.account;
  const sub = a ? `${replay ? "Replay" : `#${a.id}`} · ${cap(a.risk)} · ${a.mode === "auto" ? "Auto" : "Ask"}` : "no account yet";
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = Number(number);
    if (!Number.isInteger(id) || id < 1) {
      setProblem("type an account number");
      return;
    }
    try {
      await onSwitch(id);
      setOpen(false);
      setNumber("");
      setProblem(null);
    } catch (err) {
      setProblem(err instanceof Error ? err.message : "that account could not be opened");
    }
  };

  return (
    <div className="who-wrap" ref={wrap}>
      <button type="button" className="who" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)} aria-label={a ? `Account: ${a.name}, ${sub}` : "Account menu"}>
        <div className="avatar">{a ? initials(a.name) : "?"}</div>
        <div>
          <b>{a ? a.name : "Guest"}</b>
          <span className="num">{sub}</span>
        </div>
      </button>
      {open && (
        <div className="menu" role="menu">
          <div className="menu-h">{a ? (replay ? "A recorded session" : `Account number #${a.id}`) : "No account yet"}</div>
          {replay ? (
            <p className="hint">The account of a recorded session is read-only here.</p>
          ) : (
            <>
              <form onSubmit={submit}>
                <label className="f" htmlFor="switch-account">
                  Switch account
                </label>
                <div className="row">
                  <input id="switch-account" className="in num" inputMode="numeric" placeholder="account number" value={number} onChange={(e) => setNumber(e.target.value)} disabled={busy} />
                  <button type="submit" className="btn" disabled={busy || number.trim() === ""}>
                    Open
                  </button>
                </div>
                {problem && (
                  <p className="err" role="alert">
                    {problem}
                  </p>
                )}
              </form>
              <button
                type="button"
                className="btn ghost menu-new"
                disabled={busy}
                onClick={() => {
                  setOpen(false);
                  onNewAccount();
                }}
              >
                New account
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
