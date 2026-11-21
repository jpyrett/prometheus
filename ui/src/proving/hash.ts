// The page's address (spec 13.1): no router, the URL hash says where we are.
//   (nothing) or #desk        the Desk tab, as it always was
//   #proving                   the Proving Ground, Time Machine
//   #proving/track             the Track Record
//   #proving/live              Live
//   #proving/run/<id>          a saved run opened in the Time Machine
export type Route = { tab: "desk" } | { tab: "proving"; view: "time_machine" } | { tab: "proving"; view: "track" } | { tab: "proving"; view: "live" } | { tab: "proving"; view: "run"; runId: string };

export const DESK: Route = { tab: "desk" };

export function parseHash(hash: string): Route {
  const text = hash.replace(/^#\/?/, "");
  const [head, second, ...rest] = text.split("/");
  if (head !== "proving") return DESK;
  if (second === "track") return { tab: "proving", view: "track" };
  if (second === "live") return { tab: "proving", view: "live" };
  if (second === "run" && rest.length > 0 && rest[0]) {
    try {
      return { tab: "proving", view: "run", runId: decodeURIComponent(rest.join("/")) };
    } catch {
      return { tab: "proving", view: "time_machine" };
    }
  }
  return { tab: "proving", view: "time_machine" };
}

export function formatHash(route: Route): string {
  if (route.tab === "desk") return "";
  switch (route.view) {
    case "track":
      return "#proving/track";
    case "live":
      return "#proving/live";
    case "run":
      return `#proving/run/${encodeURIComponent(route.runId)}`;
    default:
      return "#proving";
  }
}

export function sameRoute(a: Route, b: Route): boolean {
  return formatHash(a) === formatHash(b);
}
