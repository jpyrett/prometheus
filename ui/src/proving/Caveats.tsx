// The caveats of a run (spec 13.2 (6)): a quiet strip that is always visible and never collapsed (P3), with the models and the config versions the run used.
import { Card } from "../components/ui/Card";
import type { ProvingState } from "./provingReducer";

export function Caveats({ state }: { state: ProvingState }) {
  const models = Object.entries(state.models).filter(([, v]) => typeof v === "string" || typeof v === "number");
  const versions = Object.entries(state.configVersions).filter(([, v]) => typeof v === "string" || typeof v === "number");
  return (
    <Card title="What this test can and cannot tell you" className="caveats" right={state.langsmith?.url ? <a href={state.langsmith.url} target="_blank" rel="noreferrer noopener">Trace</a> : undefined}>
      <div className="card-b">
        {state.caveats.length === 0 ? (
          <p className="muted">The server named no caveat for this run.</p>
        ) : (
          <ul className="caveat-list">
            {state.caveats.map((c) => (
              <li key={c.code} data-caveat={c.code}>
                {c.sentence}
              </li>
            ))}
          </ul>
        )}
        {(models.length > 0 || versions.length > 0) && (
          <p className="caveat-meta muted num">
            {models.map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ")}
            {models.length > 0 && versions.length > 0 ? " · " : ""}
            {versions.length > 0 ? `config versions ${versions.map(([k, v]) => `${k} ${v}`).join(", ")}` : ""}
          </p>
        )}
      </div>
    </Card>
  );
}
