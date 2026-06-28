import { useEffect, useRef } from "react";
import type { Message as MessageType } from "../types/sse";
import { Message } from "./Message";
import { DeflectCard } from "./DeflectCard";

type Props = {
  messages: MessageType[];
  onNewSession: () => void;
};

export function MessageList({ messages, onNewSession }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as new tokens arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-gray-400 text-sm select-none">
        Ask me anything to get started.
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-4">
      {messages.map((msg, i) => {
        if (msg.role === "deflect") {
          return (
            <DeflectCard
              key={i}
              message={msg.message}
              onNewSession={onNewSession}
            />
          );
        }
        return <Message key={i} message={msg} />;
      })}
      <div ref={bottomRef} />
    </div>
  );
}
