/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { DeflectCard } from "./DeflectCard";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderDeflectCard(overrides: { message?: string; onNewSession?: jest.Mock } = {}) {
  const message = overrides.message ?? "This is a deflect message.";
  const onNewSession = overrides.onNewSession ?? jest.fn();
  render(<DeflectCard message={message} onNewSession={onNewSession} />);
  return { message, onNewSession };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DeflectCard", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders the message text", () => {
    renderDeflectCard({ message: "Sorry, I can't help with that." });
    expect(screen.getByText("Sorry, I can't help with that.")).toBeTruthy();
  });

  it("renders the 'Start a new conversation' button", () => {
    renderDeflectCard();
    expect(screen.getByRole("button", { name: /start a new conversation/i })).toBeTruthy();
  });

  it("calls onNewSession when the button is clicked", () => {
    const { onNewSession } = renderDeflectCard();
    fireEvent.click(screen.getByRole("button", { name: /start a new conversation/i }));
    expect(onNewSession).toHaveBeenCalledTimes(1);
  });

  it("does not call onNewSession before any interaction", () => {
    const { onNewSession } = renderDeflectCard();
    expect(onNewSession).not.toHaveBeenCalled();
  });

  it("calls onNewSession each time the button is clicked", () => {
    const { onNewSession } = renderDeflectCard();
    const button = screen.getByRole("button", { name: /start a new conversation/i });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(onNewSession).toHaveBeenCalledTimes(2);
  });
});
