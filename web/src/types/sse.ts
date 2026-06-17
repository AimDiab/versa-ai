/**
 * SSE event shapes emitted by POST /api/chat.
 *
 * Every response — cold, warm, or deflect — streams the same event types
 * so the client only needs one handler.
 */

export type MetaEvent = {
  type: "meta";
  path: "cold" | "warm";
  session_id: string;
};

export type ChunkEvent = {
  type: "chunk";
  text: string;
};

export type DeflectEvent = {
  type: "deflect";
  message: string;
  session_id: string;
};

export type DoneEvent = {
  type: "done";
};

export type SSEEvent = MetaEvent | ChunkEvent | DeflectEvent | DoneEvent;

// ---------------------------------------------------------------------------
// Message shapes for local UI state
// ---------------------------------------------------------------------------

export type UserMessage = {
  role: "user";
  text: string;
};

export type AssistantMessage = {
  role: "assistant";
  text: string;
  /** True while tokens are still streaming in. */
  streaming: boolean;
};

export type DeflectMessage = {
  role: "deflect";
  message: string;
};

export type Message = UserMessage | AssistantMessage | DeflectMessage;
