import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listKBs, createKB } from '../api/client';
import type { KnowledgeBase, RagType } from '../types';
import KBCard from '../components/KBCard';
import RAGTypeSelector from '../components/RAGTypeSelector';

const DEFAULT_FORM = {
  name: '',
  description: '',
  rag_type: 'hybrid' as RagType,
  chunk_size: 1000,
  chunk_overlap: 200,
  top_k: 4,
  retriever_k: 10,
  graph_hops: 1,
};

export default function KBListPage() {
  const navigate = useNavigate();
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  const loadKBs = async () => {
    try {
      setKbs(await listKBs());
    } catch {
      /* silent */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadKBs(); }, []);

  const openModal = () => {
    setForm(DEFAULT_FORM);
    setError('');
    setShowAdvanced(false);
    setShowModal(true);
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) { setError('Name is required'); return; }
    setCreating(true);
    setError('');
    try {
      const kb = await createKB(form);
      setShowModal(false);
      navigate(`/kb/${kb.kb_id}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Knowledge Bases</h1>
          <p className="text-sm text-gray-500 mt-1">Create isolated document collections with different RAG pipelines</p>
        </div>
        <button
          onClick={openModal}
          className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-xl hover:bg-blue-700 transition-colors shadow-sm"
        >
          + New Knowledge Base
        </button>
      </div>

      {/* KB grid */}
      {loading ? (
        <div className="text-center py-16 text-gray-400">Loading…</div>
      ) : kbs.length === 0 ? (
        <div className="text-center py-20">
          <div className="text-5xl mb-4">📚</div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No knowledge bases yet</h2>
          <p className="text-gray-400 mb-6 text-sm">Create your first one to start uploading documents and building a RAG pipeline.</p>
          <button
            onClick={openModal}
            className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-xl hover:bg-blue-700 transition-colors"
          >
            Create Knowledge Base
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {kbs.map((kb) => (
            <KBCard key={kb.kb_id} kb={kb} onClick={() => navigate(`/kb/${kb.kb_id}`)} />
          ))}
        </div>
      )}

      {/* Create modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
              <h2 className="font-semibold text-gray-900 text-lg">Create Knowledge Base</h2>
              <button onClick={() => setShowModal(false)} className="text-gray-400 hover:text-gray-600 text-xl leading-none">&times;</button>
            </div>

            <form onSubmit={handleCreate} className="px-6 py-5 space-y-5">
              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name <span className="text-red-400">*</span></label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="e.g. Product Documentation"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={form.description}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                  placeholder="What documents will this knowledge base contain?"
                  rows={2}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* RAG Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">RAG Pipeline Type</label>
                <RAGTypeSelector
                  value={form.rag_type}
                  onChange={(v) => setForm({ ...form, rag_type: v })}
                />
              </div>

              {/* Advanced settings */}
              <div>
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  {showAdvanced ? '▲ Hide' : '▶ Show'} Advanced Settings
                </button>

                {showAdvanced && (
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    {[
                      { key: 'chunk_size', label: 'Chunk Size', min: 100, max: 4000 },
                      { key: 'chunk_overlap', label: 'Chunk Overlap', min: 0, max: 500 },
                      { key: 'top_k', label: 'Top K Results', min: 1, max: 20 },
                      { key: 'retriever_k', label: 'Retriever K', min: 1, max: 50 },
                      { key: 'graph_hops', label: 'Graph Hops', min: 1, max: 3 },
                    ].map(({ key, label, min, max }) => (
                      <div key={key}>
                        <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
                        <input
                          type="number"
                          min={min}
                          max={max}
                          value={(form as Record<string, unknown>)[key] as number}
                          onChange={(e) => setForm({ ...form, [key]: parseInt(e.target.value) || 0 })}
                          className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {error && <p className="text-sm text-red-500">{error}</p>}

              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="flex-1 px-4 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                  {creating ? 'Creating…' : 'Create Knowledge Base'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
