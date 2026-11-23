export type RightView = "news" | "workorder";

/** The two views of the right column (Jacob, 2026-11-23: "Can toggle the pane to show both"): the News of the last request that asked only for the news, and the work order with its recommendation. Each keeps its content while the other is shown. */
export function RightToggle({ view, onView }: { view: RightView; onView: (view: RightView) => void }) {
  const tab = (key: RightView, label: string) => (
    <button type="button" role="tab" aria-selected={view === key} className={`viewtab${view === key ? " on" : ""}`} onClick={() => onView(key)}>
      {label}
    </button>
  );
  return (
    <div className="viewtoggle" role="tablist" aria-label="Right column">
      {tab("news", "News")}
      {tab("workorder", "Work order")}
    </div>
  );
}
