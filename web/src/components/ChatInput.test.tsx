/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { ChatInput } from "./ChatInput";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderChatInput(overrides: { onSend?: jest.Mock; disabled?: boolean } = {}) {
  const onSend = overrides.onSend ?? jest.fn();
  const disabled = overrides.disabled ?? false;
  render(<ChatInput onSend={onSend} disabled={disabled} />);
  return { onSend };
}

function getTextarea() {
  return screen.getByRole("textbox");
}

function getSendButton() {
  return screen.getByRole("button", { name: /send/i });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ChatInput", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders the textarea and send button", () => {
    renderChatInput();
    expect(getTextarea()).toBeTruthy();
    expect(getSendButton()).toBeTruthy();
  });

  it("updates the textarea value as the user types", () => {
    renderChatInput();
    const textarea = getTextarea() as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "Hello world" } });
    expect(textarea.value).toBe("Hello world");
  });

  it("calls onSend with trimmed value and clears the input on form submit", () => {
    const { onSend } = renderChatInput();
    const textarea = getTextarea();
    fireEvent.change(textarea, { target: { value: "  Hello  " } });
    fireEvent.click(getSendButton());
    expect(onSend).toHaveBeenCalledWith("Hello");
    expect((textarea as HTMLTextAreaElement).value).toBe("");
  });

  it("calls onSend when Enter is pressed without Shift", () => {
    const { onSend } = renderChatInput();
    const textarea = getTextarea();
    fireEvent.change(textarea, { target: { value: "Enter message" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSend).toHaveBeenCalledWith("Enter message");
  });

  it("does not call onSend when Shift+Enter is pressed", () => {
    const { onSend } = renderChatInput();
    const textarea = getTextarea();
    fireEvent.change(textarea, { target: { value: "multiline" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("does not call onSend when input is blank or whitespace-only", () => {
    const { onSend } = renderChatInput();
    const textarea = getTextarea();
    fireEvent.change(textarea, { target: { value: "   " } });
    fireEvent.click(getSendButton());
    expect(onSend).not.toHaveBeenCalled();
  });

  it("does not call onSend when disabled and form is submitted", () => {
    const { onSend } = renderChatInput({ disabled: true });
    const textarea = getTextarea();
    fireEvent.change(textarea, { target: { value: "test" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables the textarea and button when disabled prop is true", () => {
    renderChatInput({ disabled: true });
    expect((getTextarea() as HTMLTextAreaElement).disabled).toBe(true);
    expect((getSendButton() as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables the send button when the textarea is empty", () => {
    renderChatInput();
    expect((getSendButton() as HTMLButtonElement).disabled).toBe(true);
  });

  it("enables the send button when the textarea has non-whitespace content", () => {
    renderChatInput();
    fireEvent.change(getTextarea(), { target: { value: "Hi" } });
    expect((getSendButton() as HTMLButtonElement).disabled).toBe(false);
  });
});
