import type { KnowledgeBase, RagType } from '../types';

const RAG_TYPE_STYLES: Record<RagType, string> = {
  vector:      'bg-blue-100 text-blue-700',
  bm25:        'bg-purple-100 text-purple-700',
  hybrid:      'bg-green-100 text-green-700',
  graph:       'bg-orange-100 text-orange-700',
  full_hybrid: 'bg-red-100 text-red-700',
};

const RAG_TYPE_LABELS: Record<RagType, string> = {
  vector:      'Vector',
  bm25:        'BM25',
  hybrid:      'Hybrid',
  graph:       'Graph',
  full_hybrid: 'Full Hybrid',
};

interface Props {
  kb: KnowledgeBase;
  onClick: () => void;
}

export default function KBCard({ kb, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-white rounded-xl border border-gray-200 shadow-sm p-5 hover:border-blue-300 hover:bg-blue-50 hover:shadow-md transition-all duration-150"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <h3 className="font-semibold text-gray-900 text-base truncate">{kb.name}</h3>
        <span className={`flex-shrink-0 text-xs font-medium px-2 py-1 rounded-full ${RAG_TYPE_STYLES[kb.rag_type]}`}>
          {RAG_TYPE_LABELS[kb.rag_type]}
        </span>
      </div>

      {kb.description && (
        <p className="text-sm text-gray-500 line-clamp-2 mb-3">{kb.description}</p>
      )}

      <div className="flex items-center gap-4 text-xs text-gray-400">
        <span>{kb.total_documents} doc{kb.total_documents !== 1 ? 's' : ''}</span>
        <span>&middot;</span>
        <span>{kb.total_chunks} chunks</span>
        <span>&middot;</span>
        <span>{new Date(kb.created_at).toLocaleDateString()}</span>
      </div>
    </button>
  );
}
