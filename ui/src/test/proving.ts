// The recorded answers and streams of the Proving Ground for the tests (made by scripts/make_proving_fixtures.mjs and scripts/record_proving.mjs from the REAL recordings of tests/fixtures/proving_real/ and the
// replay server, never typed).
import coverage from "../fixtures/proving_coverage.json";
import fastDay from "../fixtures/proving_day_oil_weekly_6m_fast_2026-06-08.json";
import liveOil from "../fixtures/proving_live_oil.json";
import livePlan from "../fixtures/proving_live_plan.json";
import plan from "../fixtures/proving_plan.json";
import planFaithful from "../fixtures/proving_plan_faithful.json";
import fastView from "../fixtures/proving_run_view_oil_weekly_6m_fast.json";
import shortView from "../fixtures/proving_run_view_oil_weekly_short.json";
import thinView from "../fixtures/proving_run_view_thin_news.json";
import runsIndex from "../fixtures/proving_runs_index.json";
import track from "../fixtures/proving_track.json";
import trackView from "../fixtures/proving_track_view.json";
import type { Coverage, DayView, LiveSnapshot, PlanAnswer, ProvingRun, RunIndexRow, TimeMachineRequest } from "../proving/types";

export const trackRun = track.data as unknown as ProvingRun;
export const liveOilSnapshot = liveOil.data as unknown as LiveSnapshot;
export const livePlanSnapshot = livePlan.data as unknown as LiveSnapshot;
export const trackViewRun = trackView.data as unknown as ProvingRun;       // GET /api/proving/track: the run view (caveats as sentences, the tiles)
export const coverageFixture = coverage.data as unknown as Coverage;     // GET /api/proving/coverage
export const planRequest = plan.request as unknown as TimeMachineRequest; // the first quick prompt, fast: the request whose research is in the cache
export const planFixture = plan.data as unknown as PlanAnswer;           // POST /api/proving/plan for planRequest
export const planFaithfulRequest = planFaithful.request as unknown as TimeMachineRequest;
export const planFaithfulFixture = planFaithful.data as unknown as PlanAnswer;

// The two real recordings of tests/fixtures/proving_real/: the fast six-month run (23 weekly decision days, buys, exits, cooldowns) and the short run (4 days, every one from the cache), each as the page's view and as
// the event stream of "Watch it again". `thinNewsRun` is a later real fast run of the same request that has decision days with thin news.
export const fastRun = fastView.data as unknown as ProvingRun;
export const shortRun = shortView.data as unknown as ProvingRun;
export const thinNewsRun = thinView.data as unknown as ProvingRun;
export const fastDayView = fastDay.data as unknown as DayView;
export const savedRunsIndex = runsIndex.data as unknown as RunIndexRow[];

// The report card of the Critic's rules as the replay server answered it (GET /api/proving/critic-report, recorded by scripts/record_proving.mjs): the saved real runs of both versions of the Critic.
import criticReport from "../fixtures/proving_critic_report.json";
import type { CriticReport } from "../proving/types";
export const criticReportFixture = criticReport.data as unknown as CriticReport;
