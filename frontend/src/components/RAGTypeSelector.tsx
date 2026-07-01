import type { RagType } from '../types';

const RAG_TYPES: { value: RagType; icon: string; label: string; desc: string }[] = [
  { value: 'vector',      icon: '⚡', label: 'Vector Only',  desc: 'Semantic search. Fast, general purpose.' },
  { value: 'bm25',        icon: '🔍', label: 'Keyword BM25', desc: 'Exact match. Best for technical docs.' },
  { value: 'hybrid',      icon: '⚖️', label: 'Hybrid',       desc: 'Vector + BM25 via RRF. Recommended.' },
  { value: 'graph',       icon: '🕸️', label: 'Graph RAG',    desc: 'Neo4j traversal. Requires Neo4j.' },
  { value: 'full_hybrid', icon: '🚀', label: 'Full Hybrid',  desc: 'All three combined. Most powerful.' },
];

interface Props {
  value: RagType;
  onChange: (v: RagType) => void;
}

export default function RAGTypeSelector({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {RAG_TYPES.map((t) => (
        <button
          key={t.value}
          type="button"
          onClick={() => onChange(t.value)}
          className={`text-left p-3 rounded-lg border-2 transition-all duration-100 ${
            value === t.value
              ? 'border-blue-600 bg-blue-50 ring-2 ring-blue-200'
              : 'border-gray-200 bg-white hover:border-blue-300 hover:bg-gray-50'
          }`}
        >
          <div className="text-xl mb-1">{t.icon}</div>
          <div className="font-semibold text-gray-900 text-sm">{t.label}</div>
          <div className="text-xs text-gray-500 mt-0.5">{t.desc}</div>
        </button>
      ))}
    </div>
  );
}
