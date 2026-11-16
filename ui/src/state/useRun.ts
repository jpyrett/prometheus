// The run of the page: an EventSource feeding runReducer (spec 5.4). On a reconnect it passes the last seq; on load an active run is attached
// and everything redraws from the replayed events.
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { connectEvents, type EventConnection } from "../api/events";
import type { RunRequest, ServerEvent } from "../api/types";
import { initialRunState, runReducer } from "./runReducer";

export function useRun() {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
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

  const start = useCallback(
    async (request: RunRequest) => {
      setError(null);
      setStarting(true);
      try {
        const { run_id } = await api.startRun(request);
        attach(run_id);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "the run could not be started");
      } finally {
        setStarting(false);
      }
    },
    [attach],
  );

  const answer = useCallback(
    async (questionId: string, value: boolean | string) => {
      if (!state.runId) return;
      try {
        await api.answer(state.runId, questionId, value);
      } catch (e) {
        // 409: the question is already closed (answered, timed out or stopped): the question_closed event is on its way.
        if (!(e instanceof ApiError && e.status === 409)) setError(e instanceof ApiError ? e.message : "the answer could not be sent");
      }
    },
    [state.runId],
  );

  const stop = useCallback(async () => {
    if (!state.runId) return;
    try {
      await api.stop(state.runId);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "the run could not be stopped");
    }
  }, [state.runId]);

  /** Feed events straight to the reducer (the fixture mode of the screenshots). */
  const feed = useCallback((events: ServerEvent[]) => {
    for (const event of events) dispatch(event);
  }, []);

  return { state, error, starting, reconnecting, start, answer, stop, attach, feed, clearError: () => setError(null) };
}
