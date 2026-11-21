// "Does each rule of the Critic earn its place?" (Critic v2 rule C1, `design/critic_v2_evidence_spec.md` section 6): the report card of the Critic's rules on the Track Record tab. One row for each finding code, a dot for the gap with its
// two-standard-error bar on a common axis, zero marked, the suitability rows greyed, a table twin. Every number and every reading comes from GET /api/proving/critic-report (`prometheus/proving/report_card.py`): the page draws, it computes nothing (P12).
// The versions of the Critic are never mixed in one row: a chip picks the version (the newest by default), another the theme (the run's own request).
import { useMemo, useState } from "react";
import { RuleGaps, type RuleGapRow } from "../charts/RuleGaps";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { points } from "../format";
import { ChipGroup } from "./Chips";
import type { CriticFamilyRow, CriticReport, CriticRuleRow, CriticVersionCard, ForwardStats } from "./types";
import { useCriticReport } from "./useProvingData";

const KIND: Record<string, string> = { blocker: "blocker", human: "asked for a person", warning: "warning", verdict: "verdict" };

/** A rule's row as the chart draws it: its pooled numbers, or its cell of one theme (null when the rule never fired in that theme). */
export function rowOf(r: CriticRuleRow, theme: string | null): RuleGapRow | null {
  const base = { key: r.code, label: r.kind === "verdict" ? "The verdict: denied against approved" : r.code, kind: r.kind, suitability: r.suitability };
  if (theme === null) {
    return {
      ...base,
      sub: r.kind === "verdict" ? `denied n=${r.hit.n}, approved n=${r.rest.n}` : `${KIND[r.kind]} · fired ${r.fired}${r.hit.closed ? `, won ${r.hit.won}/${r.hit.closed}` : ""}`,
      n: r.hit.n, fired: r.fired, closed: r.hit.closed, won: r.hit.won, hitMean: r.hit.mean_excess, restMean: r.rest.mean_excess, gap: r.gap, se: r.se, inside: r.inside_chance, reading: r.sentence_code, sentence: r.sentence,
    };
  }
  const cell = r.by_theme[theme];
  if (!cell) return null;
  return {
    ...base,
    sub: r.kind === "verdict" ? `denied n=${cell.n}, approved n=${cell.rest_n}` : `${KIND[r.kind]} · n=${cell.n}${cell.closed ? `, won ${cell.won}/${cell.closed}` : ""}`,
    n: cell.n, fired: cell.n, closed: cell.closed, won: cell.won, hitMean: cell.mean_excess, restMean: cell.rest_mean_excess, gap: cell.gap, se: cell.se, inside: cell.inside_chance, reading: cell.sentence_code, sentence: "",
  };
}

export function rowsOfVersion(card: CriticVersionCard, theme: string | null): RuleGapRow[] {
  return [card.verdict, ...card.rows].map((r) => rowOf(r, theme)).filter((r): r is RuleGapRow => r !== null);
}

export function CriticRulesCard({ report, minSample = 20 }: { report: CriticReport; minSample?: number }) {
  const versions = report.config_versions;
  const [version, setVersion] = useState<string>(versions[versions.length - 1] ?? "");
  const [theme, setTheme] = useState<string>("all");
  const card = report.by_config_version[version] ?? report.by_config_version[versions[versions.length - 1] ?? ""];
  const rows = useMemo(() => (card ? rowsOfVersion(card, theme === "all" ? null : theme) : []), [card, theme]);
  if (!card) {
    return (
      <Card title="Does each rule of the Critic earn its place?">
        <div className="empty">
          <b>No review has an outcome yet</b>The card fills in when a Time Machine run or a live session has recommendations the Critic reviewed.
        </div>
      </Card>
    );
  }
  const small = card.verdict.sentence_code === "too_few";
  return (
    <Card title="Does each rule of the Critic earn its place?" right={small ? <Badge tone="neutral" title={`A group under ${minSample} outcomes: shown with no colour of success or failure`}>SMALL SAMPLE</Badge> : null} className="criticrules">
      <div className="card-b">
        <div className="criticrules-bar">
          {versions.length > 1 && (
            <ChipGroup
              label="Version of the Critic"
              value={version}
              onChange={setVersion}
              options={versions.map((v) => ({ value: v, label: `Critic version ${v}`, count: report.by_config_version[v]?.reviewed, title: "The versions of the Critic are never mixed in one row" }))}
            />
          )}
          <ChipGroup label="Theme" value={theme} onChange={setTheme} options={[{ value: "all", label: "All themes" }, ...report.themes.map((t) => ({ value: t, label: t }))]} />
        </div>
        <RuleGaps rows={rows} band={report.chance_band} />
        <p className="q-sentence">{theme === "all" ? card.verdict.sentence : `${theme}: ${card.verdict.by_theme[theme]?.sentence_code.replace(/_/g, " ") ?? "no review"}.`}</p>
        <p className="q-note muted">
          {card.reviewed} reviewed recommendations with an outcome in version {card.config_version}; {card.suitability_excluded} denied for the user&apos;s risk profile are listed and left out of every comparison. A rule is judged beyond {report.chance_band} standard errors,
          with at least {minSample} outcomes in the smaller group. {report.limits.join(" ")}
        </p>
        {card.by_family.length > 0 && (
          <table className="tbl criticrules-families">
            <caption className="sr-only">The Critic&apos;s verdict inside each strategy family of the Advisor</caption>
            <thead>
              <tr>
                <th>Family</th>
                <th className="r">Outcomes</th>
                <th className="r">Denied</th>
                <th className="r">Approved</th>
                <th className="r" title="The stock against the benchmark 20 sessions after the decision day, whatever the plan did: all of the family, then the denied and the approved">After 20 sessions</th>
                <th>Reading</th>
              </tr>
            </thead>
            <tbody>
              {card.by_family.map((f) => (
                <tr key={f.family}>
                  <td>{f.family.replace(/_/g, " ")}</td>
                  <td className="r num">{f.n}</td>
                  <td className="r num">{`${f.denied.n} · ${points(f.denied.mean_excess)}`}</td>
                  <td className="r num">{`${f.approved.n} · ${points(f.approved.mean_excess)}`}</td>
                  <td className="r num">{forwardWords(f)}</td>
                  <td>{f.sentence_code.replace(/_/g, " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}

/** A family's forward excess after 20 sessions, as the server computed it: "+3.2 pts (n 29) · denied +1.0 (n 8) · approved +4.0 (n 21)"; a dash when the card has none. */
function forwardWords(f: CriticFamilyRow): string {
  const at = (cells: ForwardStats[] | undefined) => cells?.find((c) => c.sessions === 20);
  const say = (c: ForwardStats | undefined) => (c && c.n ? `${points(c.mean_excess_pct)} (n ${c.n})` : "—");
  const all = at(f.forward?.all);
  return all && all.n ? `${say(all)} · denied ${say(at(f.forward?.denied))} · approved ${say(at(f.forward?.approved))}` : "—";
}

/** The card on the Track Record tab: asks the server (free) and draws what it answers. */
export function CriticRules({ enabled }: { enabled: boolean }) {
  const report = useCriticReport(enabled);
  if (!report.data) {
    return (
      <Card title="Does each rule of the Critic earn its place?">
        <div className="empty">
          <b>{report.error ?? "Scoring the Critic's rules against the outcomes…"}</b>
          {report.error ? "The report card reads the saved runs and their research; it costs nothing." : ""}
        </div>
      </Card>
    );
  }
  return <CriticRulesCard report={report.data} minSample={report.data.min_sample} />;
}
