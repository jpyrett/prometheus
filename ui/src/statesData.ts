import bob from "./fixtures/bob.json";
import questions from "./fixtures/questions.json";
import type { ServerEvent } from "./api/types";

export const bobEvents = bob.events as unknown as ServerEvent[];
export const confirmQuestionEvent = questions.confirm;
export const clarifyQuestionEvent = questions.clarify;
export const approveQuestionEvent = questions.approve;
export const missingQuestionEvent = questions.missing;
