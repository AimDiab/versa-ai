type Props = {
  message: string;
  onNewSession: () => void;
};

export function DeflectCard({ message, onNewSession }: Props) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[75%] rounded-2xl rounded-bl-sm border border-amber-200 bg-amber-50 px-4 py-3 shadow-sm">
        <p className="text-sm text-amber-900 leading-relaxed">{message}</p>
        <button
          onClick={onNewSession}
          className="mt-3 text-xs font-medium text-amber-700 underline underline-offset-2 hover:text-amber-900 transition-colors"
        >
          Start a new conversation
        </button>
      </div>
    </div>
  );
}
