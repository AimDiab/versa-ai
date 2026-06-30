/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { ChatApp } from "./ChatApp";
import { useChat } from "@/hooks/useChat";
import type { Message } from "@/types/sse";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

jest.mock("@/hooks/useChat");

jest.mock("@/components/ChatInput", () => ({
  ChatInput: ({
    onSend,
    disabled,
  }: {
    onSend: (q: string) => void;
    disabled: boolean;
  }) => (
    <div data-testid="chat-input" data-disabled={String(disabled)}>
      <button onClick={() => onSend("test message")}>Send</button>
    </div>
  ),
}));

jest.mock("@/components/MessageList", () => ({
  MessageList: ({
    messages,
    onNewSession,
  }: {
    messages: Message[];
    onNewSession: () => void;
  }) => (
    <div data-testid="message-list" data-message-count={messages.length}>
      <button onClick={onNewSession}>New session from list</button>
    </div>
  ),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockUseChat = useChat as jest.MockedFunction<typeof useChat>;

function makeDefaultHook(overrides: Partial<ReturnType<typeof useChat>> = {}) {
  return {
    messages: [] as Message[],
    isStreaming: false,
    send: jest.fn(),
    newSession: jest.fn(),
    sessionId: "test-session-id",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ChatApp", () => {
  beforeEach(() => {
    mockUseChat.mockReturnValue(makeDefaultHook());
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders the app title in the header", () => {
    render(<ChatApp />);
    expect(screen.getByText("Versa AI")).toBeTruthy();
  });

  it("renders the New conversation button", () => {
    render(<ChatApp />);
    expect(screen.getByText("New conversation")).toBeTruthy();
  });

  it("renders the MessageList and ChatInput", () => {
    render(<ChatApp />);
    expect(screen.getByTestId("message-list")).toBeTruthy();
    expect(screen.getByTestId("chat-input")).toBeTruthy();
  });

  it("calls newSession when New conversation button is clicked", () => {
    const newSession = jest.fn();
    mockUseChat.mockReturnValue(makeDefaultHook({ newSession }));

    render(<ChatApp />);
    fireEvent.click(screen.getByText("New conversation"));

    expect(newSession).toHaveBeenCalledTimes(1);
  });

  it("passes newSession to MessageList as onNewSession", () => {
    const newSession = jest.fn();
    mockUseChat.mockReturnValue(makeDefaultHook({ newSession }));

    render(<ChatApp />);
    fireEvent.click(screen.getByText("New session from list"));

    expect(newSession).toHaveBeenCalledTimes(1);
  });

  it("passes the current messages to MessageList", () => {
    const messages: Message[] = [
      { role: "user", text: "hello" },
      { role: "assistant", text: "hi there", streaming: false },
    ];
    mockUseChat.mockReturnValue(makeDefaultHook({ messages }));

    render(<ChatApp />);

    const list = screen.getByTestId("message-list");
    expect(list.getAttribute("data-message-count")).toBe("2");
  });

  it("forwards send to ChatInput", () => {
    const send = jest.fn();
    mockUseChat.mockReturnValue(makeDefaultHook({ send }));

    render(<ChatApp />);
    fireEvent.click(screen.getByText("Send"));

    expect(send).toHaveBeenCalledWith("test message");
  });

  it("disables ChatInput while streaming", () => {
    mockUseChat.mockReturnValue(makeDefaultHook({ isStreaming: true }));

    render(<ChatApp />);

    expect(screen.getByTestId("chat-input").getAttribute("data-disabled")).toBe(
      "true"
    );
  });

  it("enables ChatInput when not streaming", () => {
    mockUseChat.mockReturnValue(makeDefaultHook({ isStreaming: false }));

    render(<ChatApp />);

    expect(screen.getByTestId("chat-input").getAttribute("data-disabled")).toBe(
      "false"
    );
  });
});
