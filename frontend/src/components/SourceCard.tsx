import { useState } from 'react';
import type { Source } from '../types';

interface Props {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: Props) {
  const [open, setOpen] = useState(false);
  const { name, chunk_index, graph_relevance } = source.metadata;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex justify-between items-center px-3 py-2 bg-gray-50 hover:bg-gray-100 text-left transition-colors"
      >
        <span className="text-gray-500 truncate pr-2">
          <span className="text-gray-400 mr-1">[{index}]</span>
          <span className="text-gray-700 font-medium">{name}</span>
          <span className="text-gray-400"> — chunk {chunk_index}</span>
          {graph_relevance !== undefined && (
            <span className="ml-2 text-purple-500 font-mono">graph:{graph_relevance}</span>
          )}
        </span>
        <span className="text-gray-400 flex-shrink-0">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-3 py-2 bg-white text-gray-600 leading-relaxed border-t border-gray-100 whitespace-pre-wrap">
          {source.content}
        </div>
      )}
    </div>
  );
}
