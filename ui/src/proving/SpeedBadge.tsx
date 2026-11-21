// A run's speed as a small badge (fast or faithful), beside its headline and on the saved-run cards. Grey: a speed is a fact about how the run was made, not a result. A run saved before the speeds has none, and shows none.
import { Badge } from "../components/ui/Badge";
import { SPEED_TITLE, SPEED_WORDS } from "./timing";
import type { Speed } from "./types";

export function SpeedBadge({ speed }: { speed: Speed | null | undefined }) {
  if (!speed) return null;
  return (
    <Badge tone="neutral" title={SPEED_TITLE[speed]}>
      {SPEED_WORDS[speed]}
    </Badge>
  );
}
