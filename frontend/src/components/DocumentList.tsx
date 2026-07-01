import type { Document } from '../types';

interface Props {
  documents: Document[];
  onRefresh: () => void;
}

function formatBytes(bytes: number): string {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const TYPE_COLORS: Record<string, string> = {
  pdf:  'bg-red-100 text-red-600',
  txt:  'bg-blue-100 text-blue-600',
  md:   'bg-purple-100 text-purple-600',
  text: 'bg-gray-100 text-gray-500',
};

export default function DocumentList({ documents, onRefresh }: Props) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Documents ({documents.length})
        </h2>
        <button
          onClick={onRefresh}
          className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
          title="Refresh document list"
        >
          ↻
        </button>
      </div>

      {documents.length === 0 ? (
        <p className="text-xs text-gray-400 text-center py-4">
          No documents yet. Upload a file above.
        </p>
      ) : (
        <ul className="space-y-2">
          {documents.map((doc, i) => (
            <li
              key={`${doc.document_name}-${i}`}
              className="bg-white rounded-lg p-3 border border-gray-100 text-xs"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="font-medium text-gray-800 truncate flex-1">
                  {doc.document_name}
                </span>
                <span className={`flex-shrink-0 px-1.5 py-0.5 rounded text-xs font-mono uppercase ${
                  TYPE_COLORS[doc.document_type] ?? 'bg-gray-100 text-gray-500'
                }`}>
                  {doc.document_type}
                </span>
              </div>
              <div className="text-gray-400 mt-1 flex items-center gap-2">
                <span>{doc.num_chunks} chunks</span>
                <span>·</span>
                <span>{formatBytes(doc.file_size)}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
