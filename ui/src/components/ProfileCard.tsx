import { useEffect, useId, useRef, useState } from "react";
import { ApiError, type NewAccount } from "../api/client";
import type { AccountView, RiskLevel, TradeMode } from "../api/types";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";
import { RiskSlider } from "./ui/RiskSlider";

export interface ProfileActions {
  create: (form: NewAccount) => Promise<unknown>;
  patch: (body: Partial<{ name: string; age: number; risk: RiskLevel; mode: TradeMode }>) => Promise<unknown>;
  funds: (action: "add" | "remove", amount: string) => Promise<unknown>;
}

function message(error: unknown): string {
  return error instanceof ApiError || error instanceof Error ? error.message : "that did not work";
}

/** The profile of the account (spec 5.3 (4)): name, age, risk, funds. The trade mode is the request card's Mode switch, not this card's. Without an account the same card opens one, in the mode that switch shows (`newMode`). */
export function ProfileCard({ view, running, readOnlyReason, loadError, actions, forceOpen = false, onCancel, newMode }: { view: AccountView | null; running: boolean; readOnlyReason: string | null; loadError: string | null; actions: ProfileActions; forceOpen?: boolean; onCancel?: () => void; newMode: TradeMode }) {
  const open = view === null || forceOpen;
  const locked = running || readOnlyReason !== null;
  const reason = readOnlyReason ?? (running ? "a run is active: wait for it to finish" : undefined);
  return open ? (
    <Card title="Profile" className="profile" right="open an account">
      <OpenAccount create={actions.create} loadError={loadError} onCancel={view !== null ? onCancel : undefined} mode={newMode} />
    </Card>
  ) : (
    <Card title="Profile" className="profile" right={view ? <>account <span className="num">{view.account.id === 0 ? "replay" : `#${view.account.id}`}</span></> : undefined}>
      <div className="card-b" title={reason}>
        <Existing view={view} locked={locked} actions={actions} />
      </div>
    </Card>
  );
}

function useSaved() {
  const [saved, setSaved] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), []);
  return {
    saved,
    flash(field: string) {
      setSaved(field);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setSaved(null), 1800);
    },
  };
}

function Saved({ show }: { show: boolean }) {
  return (
    <span className={`saved${show ? " on" : ""}`} aria-live="polite">
      {show ? "✓ Saved" : ""}
    </span>
  );
}

function Existing({ view, locked, actions }: { view: AccountView; locked: boolean; actions: ProfileActions }) {
  const a = view.account;
  const id = useId();
  const [name, setName] = useState(a.name);
  const [age, setAge] = useState(a.age === null ? "" : String(a.age));
  const [risk, setRisk] = useState<RiskLevel>(a.risk);
  const [amount, setAmount] = useState("");
  const [problem, setProblem] = useState<{ field: string; text: string } | null>(null);
  const { saved, flash } = useSaved();

  // The account is the truth: what the server holds replaces what was typed (a refusal puts the old value back).
  useEffect(() => setName(a.name), [a.name]);
  useEffect(() => setAge(a.age === null ? "" : String(a.age)), [a.age]);
  useEffect(() => setRisk(a.risk), [a.risk]);

  async function save(field: string, body: Parameters<ProfileActions["patch"]>[0], undo: () => void) {
    setProblem(null);
    try {
      await actions.patch(body);
      flash(field);
    } catch (e) {
      setProblem({ field, text: message(e) });
      undo();
    }
  }

  const blurName = () => {
    const next = name.trim();
    if (next === a.name) return;
    if (!next) return setName(a.name);
    void save("name", { name: next }, () => setName(a.name));
  };
  const blurAge = () => {
    if (age.trim() === (a.age === null ? "" : String(a.age))) return;
    const n = Number(age);
    if (!Number.isInteger(n) || n < 18 || n > 120) {
      setProblem({ field: "age", text: "age must be a whole number from 18 to 120" });
      return setAge(a.age === null ? "" : String(a.age));
    }
    void save("age", { age: n }, () => setAge(a.age === null ? "" : String(a.age)));
  };
  const commitRisk = (value: RiskLevel) => {
    if (value !== a.risk) void save("risk", { risk: value }, () => setRisk(a.risk));
  };
  const money = async (action: "add" | "remove") => {
    setProblem(null);
    const text = amount.trim().replace(/^\$/, "").replace(/,/g, "");
    if (!text) return setProblem({ field: "funds", text: "type an amount first" });
    try {
      await actions.funds(action, text);
      setAmount("");
      flash("funds");
    } catch (e) {
      setProblem({ field: "funds", text: message(e) });
    }
  };
  const fail = (field: string) =>
    problem?.field === field ? (
      <p className="err" role="alert">
        {problem.text}
      </p>
    ) : null;

  return (
    <fieldset className="plain" disabled={locked}>
      <div className="row field">
        <div style={{ flex: 2.2 }}>
          <label className="f" htmlFor={`${id}-name`}>
            Name <Saved show={saved === "name"} />
          </label>
          <input className="in" id={`${id}-name`} value={name} maxLength={60} onChange={(e) => setName(e.target.value)} onBlur={blurName} onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
        </div>
        <div>
          <label className="f" htmlFor={`${id}-age`}>
            Age <Saved show={saved === "age"} />
          </label>
          <input className="in num" id={`${id}-age`} value={age} inputMode="numeric" onChange={(e) => setAge(e.target.value.replace(/[^\d]/g, ""))} onBlur={blurAge} onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
        </div>
      </div>
      {fail("name")}
      {fail("age")}
      <div className="field risk-field">
        <label className="f" htmlFor={`${id}-risk`}>
          Risk level <Saved show={saved === "risk"} />
        </label>
        <RiskSlider id={`${id}-risk`} value={risk} onChange={setRisk} onCommit={commitRisk} disabled={locked} />
        {fail("risk")}
      </div>
      <div className="field" style={{ marginBottom: 0 }}>
        <label className="f" htmlFor={`${id}-funds`}>
          Funds <Saved show={saved === "funds"} />
        </label>
        <div className="row">
          <div className="money" style={{ flex: 1.4 }}>
            <input className="in" id={`${id}-funds`} placeholder="0.00" inputMode="decimal" value={amount} onChange={(e) => setAmount(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void money("add")} />
          </div>
          <Button onClick={() => void money("add")}>Add</Button>
          <Button onClick={() => void money("remove")}>Remove</Button>
        </div>
        {fail("funds")}
        <p className="hint">Simulated money. No bank is connected.</p>
      </div>
    </fieldset>
  );
}

function OpenAccount({ create, loadError, onCancel, mode }: { create: ProfileActions["create"]; loadError: string | null; onCancel?: () => void; mode: TradeMode }) {
  const id = useId();
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [risk, setRisk] = useState<RiskLevel>("moderate");
  const [cash, setCash] = useState("10000.00");
  const [problem, setProblem] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    const years = Number(age);
    if (!name.trim()) return setProblem("type a name");
    if (!Number.isInteger(years) || years < 18 || years > 120) return setProblem("age must be a whole number from 18 to 120");
    setSaving(true);
    try {
      await create({ name: name.trim(), age: years, risk, mode, cash: cash.trim().replace(/^\$/, "").replace(/,/g, "") });
    } catch (err) {
      setProblem(message(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="card-b" onSubmit={submit}>
      {loadError && (
        <p className="err" role="alert">
          {loadError}
        </p>
      )}
      <div className="row field">
        <div style={{ flex: 2.2 }}>
          <label className="f" htmlFor={`${id}-name`}>
            Name
          </label>
          <input className="in" id={`${id}-name`} value={name} maxLength={60} autoComplete="off" onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <label className="f" htmlFor={`${id}-age`}>
            Age
          </label>
          <input className="in num" id={`${id}-age`} value={age} inputMode="numeric" autoComplete="off" onChange={(e) => setAge(e.target.value.replace(/[^\d]/g, ""))} />
        </div>
      </div>
      <div className="field risk-field">
        <label className="f" htmlFor={`${id}-risk`}>
          Risk level
        </label>
        <RiskSlider id={`${id}-risk`} value={risk} onChange={setRisk} onCommit={setRisk} />
      </div>
      <div className="field">
        <label className="f" htmlFor={`${id}-cash`}>
          Starting funds
        </label>
        <div className="money">
          <input className="in" id={`${id}-cash`} inputMode="decimal" value={cash} onChange={(e) => setCash(e.target.value)} />
        </div>
        <p className="hint">Simulated money. No bank is connected.</p>
      </div>
      {problem && (
        <p className="err" role="alert">
          {problem}
        </p>
      )}
      <div className="row">
        <Button type="submit" variant="yes" className={onCancel ? undefined : "wide"} disabled={saving}>
          Create account
        </Button>
        {onCancel && (
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}
