/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { MessageList } from "./MessageList";
import type { Message } from "../types/sse";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

jest.mock("./Message", () => ({
  Message: ({ message }: { message: { role: string; text: string } }) => (
    <div data-testid="message" data-role={message.role}>
      {message.text}
    </div>
  ),
}));

jest.mock("./DeflectCard", () => ({
  DeflectCard: ({
    message,
    onNewSession,
  }: {
    message: string;
    onNewSession: () => void;
  }) => (
    <div data-testid="deflect-card">
      <span>{message}</span>
      <button onClick={onNewSession}>New session</button>
    </div>
  ),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderList(messages: Message[], onNewSession = jest.fn()) {
  render(<MessageList messages={messages} onNewSession={onNewSession} />);
  return { onNewSession };
}

const userMsg = (text: string): Message => ({ role: "user", text });
const assistantMsg = (text: string): Message => ({
  role: "assistant",
  text,
  streaming: false,
});
const deflectMsg = (message: string): Message => ({ role: "deflect", message });

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MessageList", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("empty state", () => {
    it("shows the empty-state prompt when there are no messages", () => {
      renderList([]);
      expect(screen.getByText("Ask me anything to get started.")).toBeTruthy();
    });

    it("renders no Message or DeflectCard components when empty", () => {
      renderList([]);
      expect(screen.queryByTestId("message")).toBeNull();
      expect(screen.queryByTestId("deflect-card")).toBeNull();
    });
  });

  describe("user and assistant messages", () => {
    it("renders a Message for each user/assistant message", () => {
      renderList([userMsg("Hello"), assistantMsg("Hi there")]);
      expect(screen.getAllByTestId("message")).toHaveLength(2);
    });

    it("renders message text", () => {
      renderList([userMsg("What is AI?")]);
      expect(screen.getByText("What is AI?")).toBeTruthy();
    });

    it("passes the correct role to each Message", () => {
      renderList([userMsg("Hey"), assistantMsg("Hello")]);
      const messages = screen.getAllByTestId("message");
      expect(messages[0].getAttribute("data-role")).toBe("user");
      expect(messages[1].getAttribute("data-role")).toBe("assistant");
    });

    it("does not render the empty-state prompt when messages exist", () => {
      renderList([userMsg("Hello")]);
      expect(
        screen.queryByText("Ask me anything to get started.")
      ).toBeNull();
    });
  });

  describe("deflect messages", () => {
    it("renders a DeflectCard for deflect messages", () => {
      renderList([deflectMsg("Sorry, out of scope.")]);
      expect(screen.getByTestId("deflect-card")).toBeTruthy();
    });

    it("renders the deflect message text inside DeflectCard", () => {
      renderList([deflectMsg("I can't help with that.")]);
      expect(screen.getByText("I can't help with that.")).toBeTruthy();
    });

    it("does not render a Message component for deflect messages", () => {
      renderList([deflectMsg("Out of scope.")]);
      expect(screen.queryByTestId("message")).toBeNull();
    });

    it("calls onNewSession when DeflectCard triggers it", () => {
      const onNewSession = jest.fn();
      renderList([deflectMsg("Out of scope.")], onNewSession);
      fireEvent.click(screen.getByText("New session"));
      expect(onNewSession).toHaveBeenCalledTimes(1);
    });
  });

  describe("mixed messages", () => {
    it("renders Message and DeflectCard components in the correct order", () => {
      renderList([
        userMsg("Hello"),
        deflectMsg("Out of scope."),
        assistantMsg("Back on topic."),
      ]);
      expect(screen.getAllByTestId("message")).toHaveLength(2);
      expect(screen.getAllByTestId("deflect-card")).toHaveLength(1);
    });

    it("renders multiple deflect cards when there are multiple deflect messages", () => {
      renderList([deflectMsg("First deflect"), deflectMsg("Second deflect")]);
      expect(screen.getAllByTestId("deflect-card")).toHaveLength(2);
    });
  });
});
