import type { Message } from '../types';
import SourceCard from './SourceCard';

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`${isUser ? 'max-w-[70%]' : 'w-full max-w-[90%]'}`}>
        <p className={`text-xs text-gray-400 mb-1 ${isUser ? 'text-right' : 'text-left'}`}>
          {isUser ? 'You' : 'RAG Assistant'}
        </p>

        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-sm'
              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm'
          }`}
        >
          {message.content}
          {message.streaming && (
            <span className="inline-block w-0.5 h-4 bg-gray-400 ml-1 align-text-bottom animate-blink" />
          )}
        </div>

        {!isUser && !message.streaming && message.execution_time !== undefined && (
          <p className="text-xs text-gray-400 mt-1.5 ml-1">
            {message.execution_time.toFixed(2)}s
            {message.sources.length > 0 && (
              <span> &middot; {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}</span>
            )}
          </p>
        )}

        {!isUser && !message.streaming && message.sources.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {message.sources.map((src, i) => (
              <SourceCard key={`${src.metadata.chunk_id}-${i}`} source={src} index={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
