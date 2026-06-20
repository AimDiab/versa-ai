/**
 * useChat — manages session state and SSE streaming.
 *
 * Responsibilities:
 *   - Hold the message history for the active session
 *   - Send queries to POST /api/chat
 *   - Parse the SSE stream and append tokens to the current assistant message
 *   - Expose a `newSession()` reset so the deflect card can offer a fresh start
 */

import { useCallback, useRef, useState } from "react";
import type { Message, SSEEvent } from "../types/sse";
import { apiUrl } from "../lib/api";

function generateSessionId(): string {
  return crypto.randomUUID();
}

/** Parse a raw SSE response body into typed events. */
async function* parseSSE(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<SSEEvent> {
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";

    for (const block of lines) {
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            yield JSON.parse(line.slice("data: ".length)) as SSEEvent;
          } catch {
            // malformed event — skip
          }
        }
      }
    }
  }
}

export function useChat() {
  const [sessionId, setSessionId] = useState<string>(generateSessionId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(
    async (query: string) => {
      if (isStreaming) return;

      // Append the user message immediately
      setMessages((prev) => [...prev, { role: "user", text: query }]);

      // Placeholder for the assistant reply — will be filled as tokens arrive
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "", streaming: true },
      ]);

      setIsStreaming(true);
      abortRef.current = new AbortController();

      try {
        const response = await fetch(apiUrl("/api/chat"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, query }),
          signal: abortRef.current.signal,
        });

        if (!response.ok || !response.body) {
          throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body.getReader();

        for await (const event of parseSSE(reader)) {
          switch (event.type) {
            case "chunk":
              // Append token to the last (assistant) message
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last.role === "assistant") {
                  next[next.length - 1] = {
                    ...last,
                    text: last.text + event.text,
                  };
                }
                return next;
              });
              break;

            case "deflect":
              // Replace the placeholder assistant message with a deflect card
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  role: "deflect",
                  message: event.message,
                };
                return next;
              });
              break;

            case "done":
              // Mark streaming complete
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last.role === "assistant") {
                  next[next.length - 1] = { ...last, streaming: false };
                }
                return next;
              });
              break;

            case "meta":
              // No UI action needed — could use path for a loading indicator later
              break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last.role === "assistant") {
              next[next.length - 1] = {
                ...last,
                text: "Something went wrong. Please try again.",
                streaming: false,
              };
            }
            return next;
          });
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming, sessionId]
  );

  const newSession = useCallback(() => {
    abortRef.current?.abort();
    setSessionId(generateSessionId());
    setMessages([]);
    setIsStreaming(false);
  }, []);

  return { messages, isStreaming, send, newSession, sessionId };
}
