import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { streamKBChat } from '../api/client';
import type { Message, StreamEvent } from '../types';
import MessageBubble from './MessageBubble';

interface Props {
  kbId: string;
  sessionId: string;
}

export default function ChatInterface({ kbId, sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setMessages([]);
  }, [kbId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput('');
    setIsStreaming(true);

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      sources: [],
      timestamp: new Date(),
    };

    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      timestamp: new Date(),
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      for await (const event of streamKBChat(kbId, question, sessionId)) {
        handleStreamEvent(event, assistantId);
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Connection error: ${String(err)}`, streaming: false }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  };

  const handleStreamEvent = (event: StreamEvent, assistantId: string) => {
    if (event.type === 'token' && event.token) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + event.token! } : m,
        ),
      );
    } else if (event.type === 'done') {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, sources: event.sources ?? [], execution_time: event.execution_time, streaming: false }
            : m,
        ),
      );
    } else if (event.type === 'error') {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Error: ${event.message}`, streaming: false }
            : m,
        ),
      );
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-2xl mb-4">
              🧠
            </div>
            <h2 className="text-xl font-semibold text-gray-700 mb-2">Ask a question</h2>
            <p className="text-sm text-gray-400 max-w-sm">
              Upload documents first, then ask questions about their content.
              Powered by Groq Llama 3.3 70B.
            </p>
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-100 bg-white px-4 py-4">
        <div className="flex gap-3 items-end max-w-4xl mx-auto">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send · Shift+Enter for newline)"
            rows={2}
            disabled={isStreaming}
            className="flex-1 resize-none bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 transition-colors disabled:opacity-60"
          />
          <div className="flex flex-col gap-2 pb-0.5">
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-medium text-white transition-colors"
            >
              {isStreaming ? (
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  …
                </span>
              ) : (
                'Send'
              )}
            </button>
            {messages.length > 0 && !isStreaming && (
              <button
                onClick={() => setMessages([])}
                className="px-5 py-1.5 text-xs text-gray-400 hover:text-gray-600 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
