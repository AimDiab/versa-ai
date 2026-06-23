/**
 * @jest-environment jsdom
 */

import { act, renderHook } from "@testing-library/react";
import { useChat } from "./useChat";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a ReadableStream that emits SSE-formatted text for each event. */
function makeSSEStream(events: object[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
        );
      }
      controller.close();
    },
  });
}

/** Set up global.fetch to return a successful SSE response. */
function mockFetch(events: object[]) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    body: makeSSEStream(events),
  });
}

/** Set up global.fetch to return an HTTP error. */
function mockFetchError(status = 500) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status,
    body: null,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useChat", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("starts with empty messages and not streaming", () => {
    const { result } = renderHook(() => useChat());

    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
    expect(typeof result.current.sessionId).toBe("string");
  });

  it("appends the user message and an assistant placeholder immediately on send", async () => {
    mockFetch([{ type: "done" }]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("hello");
    });

    // After completion there should be two messages: user + assistant
    expect(result.current.messages[0]).toEqual({ role: "user", text: "hello" });
    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      streaming: false,
    });
  });

  it("accumulates chunk events into the assistant message text", async () => {
    mockFetch([
      { type: "chunk", text: "Hello" },
      { type: "chunk", text: ", world" },
      { type: "chunk", text: "!" },
      { type: "done" },
    ]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("hi");
    });

    const assistant = result.current.messages[1];
    expect(assistant).toMatchObject({
      role: "assistant",
      text: "Hello, world!",
      streaming: false,
    });
  });

  it("marks the assistant message streaming=false on done", async () => {
    mockFetch([
      { type: "chunk", text: "ok" },
      { type: "done" },
    ]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("test");
    });

    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      streaming: false,
    });
    expect(result.current.isStreaming).toBe(false);
  });

  it("replaces the assistant placeholder with a deflect message on deflect", async () => {
    mockFetch([
      { type: "deflect", message: "Please contact support.", session_id: "s1" },
    ]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("sensitive question");
    });

    expect(result.current.messages[1]).toEqual({
      role: "deflect",
      message: "Please contact support.",
    });
  });

  it("shows an error message on non-ok HTTP response", async () => {
    mockFetchError(503);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("test");
    });

    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      text: "Something went wrong. Please try again.",
      streaming: false,
    });
    expect(result.current.isStreaming).toBe(false);
  });

  it("ignores unrecognised meta events without crashing", async () => {
    mockFetch([
      { type: "meta", path: "cold", session_id: "s1" },
      { type: "chunk", text: "fine" },
      { type: "done" },
    ]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("test");
    });

    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      text: "fine",
    });
  });

  it("does not send while already streaming", async () => {
    // Never-resolving fetch so isStreaming stays true
    global.fetch = jest.fn().mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useChat());

    // Start a send but don't await — leave it in flight
    act(() => {
      result.current.send("first");
    });

    await act(async () => {
      await result.current.send("second");
    });

    // fetch should only have been called once
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it("newSession clears messages and generates a new session id", async () => {
    mockFetch([{ type: "chunk", text: "hi" }, { type: "done" }]);

    const { result } = renderHook(() => useChat());
    const originalSessionId = result.current.sessionId;

    await act(async () => {
      await result.current.send("hello");
    });

    expect(result.current.messages).toHaveLength(2);

    act(() => {
      result.current.newSession();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.sessionId).not.toBe(originalSessionId);
  });

  it("aborts the in-flight request when the component unmounts", async () => {
    const abortSpy = jest.fn();
    const abortController = { abort: abortSpy, signal: new AbortController().signal };
    jest.spyOn(global, "AbortController" as never).mockImplementation(() => abortController as never);
    global.fetch = jest.fn().mockReturnValue(new Promise(() => {}));

    const { result, unmount } = renderHook(() => useChat());

    act(() => {
      result.current.send("hello");
    });

    unmount();

    expect(abortSpy).toHaveBeenCalled();
  });

  it("sends the session_id and query in the POST body", async () => {
    mockFetch([{ type: "done" }]);

    const { result } = renderHook(() => useChat());
    const sessionId = result.current.sessionId;

    await act(async () => {
      await result.current.send("what is the return policy?");
    });

    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          query: "what is the return policy?",
        }),
      })
    );
  });
});
