/**
 * @jest-environment jsdom
 */

import { render, screen } from "@testing-library/react";
import { Message } from "./Message";
import type { UserMessage, AssistantMessage } from "../types/sse";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderMessage(message: UserMessage | AssistantMessage) {
  const { container } = render(<Message message={message} />);
  return { container };
}

function userMessage(text: string): UserMessage {
  return { role: "user", text };
}

function assistantMessage(text: string, streaming = false): AssistantMessage {
  return { role: "assistant", text, streaming };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Message", () => {
  describe("user message", () => {
    it("renders the message text", () => {
      renderMessage(userMessage("Hello there"));
      expect(screen.getByText("Hello there")).toBeTruthy();
    });

    it("does not render the streaming cursor", () => {
      const { container } = renderMessage(userMessage("Hi"));
      const cursor = container.querySelector("span");
      expect(cursor).toBeNull();
    });
  });

  describe("assistant message", () => {
    it("renders the message text", () => {
      renderMessage(assistantMessage("I can help with that."));
      expect(screen.getByText("I can help with that.")).toBeTruthy();
    });

    it("shows the streaming cursor when streaming is true", () => {
      const { container } = renderMessage(assistantMessage("Typing…", true));
      const cursor = container.querySelector("span");
      expect(cursor).toBeTruthy();
      expect(cursor!.className).toContain("animate-pulse");
    });

    it("hides the streaming cursor when streaming is false", () => {
      const { container } = renderMessage(assistantMessage("Done", false));
      const cursor = container.querySelector("span");
      expect(cursor).toBeNull();
    });
  });
});
