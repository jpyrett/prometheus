// The Time Machine's run in the page: an EventSource feeding provingReducer (the same transport as the Desk's run, `connectEvents`), plus the calls that start one, play a saved one again or open one.
// On a reconnect the last seq goes back; a run already going on when the page loads is attached by the App.
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { connectEvents, type EventConnection } from "../api/events";
import type { ServerEvent } from "../api/types";
import { initialProvingState, provingReducer } from "./provingReducer";
import type { TimeMachineRequest } from "./types";

export function useProving() {
  const [state, dispatch] = useReducer(provingReducer, initialProvingState);
  const connection = useRef<EventConnection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);

  const attach = useCallback((runId: string, after = 0) => {
    connection.current?.close();
    connection.current = connectEvents(runId, {
      after,
      onEvent: (event) => {
        setReconnecting(false);
        dispatch(event);
      },
      onReconnect: () => setReconnecting(true),
    });
  }, []);
  useEffect(() => () => connection.current?.close(), []);

  const launch = useCallback(
    async (request: TimeMachineRequest) => {
      setError(null);
      setStarting(true);
      try {
        const { run_id } = await api.startProving(request);
        dispatch({ type: "reset" });
        attach(run_id);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "the test could not be started");
      } finally {
        setStarting(false);
      }
    },
    [attach],
  );

  /** "Watch it again": the saved run played through the same events (free). */
  const watch = useCallback(
    async (id: string) => {
      setError(null);
      setStarting(true);
      try {
        const { run_id } = await api.replayProving(id);
        dispatch({ type: "reset" });
        attach(run_id);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "the run could not be played again");
      } finally {
        setStarting(false);
      }
    },
    [attach],
  );

  /** A saved run opened as it is (nothing is played). */
  const open = useCallback(async (id: string) => {
    setError(null);
    connection.current?.close();
    try {
      const run = await api.provingRun(id);
      dispatch({ type: "load", run });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "the run could not be opened");
    }
  }, []);

  const stop = useCallback(async () => {
    if (!state.runId) return;
    try {
      await api.stop(state.runId);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "the run could not be stopped");
    }
  }, [state.runId]);

  // A run that finished on the stream: its saved record carries what the stream does not (the models, the config versions, the tiles, the days that have a desk).
  const { phase, streamed, savedId, models } = state;
  useEffect(() => {
    if (phase !== "finished" || !streamed || !savedId || Object.keys(models).length > 0) return;
    let live = true;
    api
      .provingRun(savedId)
      .then((run) => live && dispatch({ type: "details", run }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [phase, streamed, savedId, models]);

  const feed = useCallback((events: ServerEvent[]) => {
    for (const event of events) dispatch(event);
  }, []);
  const reset = useCallback(() => {
    connection.current?.close();
    dispatch({ type: "reset" });
  }, []);

  return { state, error, starting, reconnecting, launch, watch, open, stop, attach, feed, reset, dispatch, clearError: () => setError(null) };
}
