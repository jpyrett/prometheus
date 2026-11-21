// The two REAL event streams of "Watch it again" (scripts/record_proving.mjs, from the replay server playing tests/fixtures/proving_real/oil_weekly_6m_fast.json and oil_weekly_short.json), for the tests only: kept out of
// test/proving.ts so that the states sheet, which is part of the build, does not carry them.
import fastStreamFile from "../fixtures/proving_stream_oil_weekly_6m_fast.json";
import shortStreamFile from "../fixtures/proving_stream_oil_weekly_short.json";
import type { ServerEvent } from "../api/types";

export const fastEvents = fastStreamFile.events as unknown as ServerEvent[];
export const shortEvents = shortStreamFile.events as unknown as ServerEvent[];
