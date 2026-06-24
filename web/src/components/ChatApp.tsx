"use client";

import { useChat } from "@/hooks/useChat";

export function ChatApp() {
  const { messages, isStreaming, send, newSession } = useChat();

  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-4 shadow-sm">
        <h1 className="text-lg font-semibold text-gray-900">Versa AI</h1>
        <button
          onClick={newSession}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-100"
        >
          New conversation
        </button>
      </header>
    </div>
  );
}
