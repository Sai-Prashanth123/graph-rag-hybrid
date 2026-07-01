import { useEffect, useState } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { getKB, updateKB, deleteKB, fetchKBDocuments, uploadFileToKB } from '../api/client';
import type { KnowledgeBase, Document, RagType } from '../types';
import ChatInterface from '../components/ChatInterface';
import FileUpload from '../components/FileUpload';
import DocumentList from '../components/DocumentList';
import APIKeysPanel from '../components/APIKeysPanel';
import RAGTypeSelector from '../components/RAGTypeSelector';

type Tab = 'documents' | 'chat' | 'apikeys' | 'settings';

function getOrCreateSessionId(): string {
  const key = 'rag_session_id';
  const stored = localStorage.getItem(key);
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

export default function KBDetailPage() {
  const { kbId } = useParams<{ kbId: string }>();
  const navigate = useNavigate();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [tab, setTab] = useState<Tab>('documents');
  const [sessionId] = useState(getOrCreateSessionId);
  const [loading, setLoading] = useState(true);

  const [settingsForm, setSettingsForm] = useState<{
    name?: string; description?: string; rag_type?: RagType;
    chunk_size?: number; chunk_overlap?: number; top_k?: number;
    retriever_k?: number; graph_hops?: number;
  }>({});
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');
  const [deleting, setDeleting] = useState(false);

  const loadKB = async () => {
    if (!kbId) return;
    try {
      const k = await getKB(kbId);
      setKb(k);
      setSettingsForm({
        name: k.name,
        description: k.description ?? undefined,
        rag_type: k.rag_type,
        chunk_size: k.chunk_size,
        chunk_overlap: k.chunk_overlap,
        top_k: k.top_k,
        retriever_k: k.retriever_k,
        graph_hops: k.graph_hops,
      });
    } catch {
      navigate('/');
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async () => {
    if (!kbId) return;
    try {
      setDocuments(await fetchKBDocuments(kbId));
    } catch { /* silent */ }
  };

  useEffect(() => { loadKB(); loadDocuments(); }, [kbId]);

  const handleSaveSettings = async () => {
    if (!kbId) return;
    setSaving(true);
    setSaveMsg('');
    try {
      const updated = await updateKB(kbId, settingsForm);
      setKb(updated);
      setSaveMsg('Settings saved.');
    } catch {
      setSaveMsg('Failed to save settings.');
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(''), 3000);
    }
  };

  const handleDeleteKB = async () => {
    if (!kbId || !kb) return;
    if (!confirm(`Delete "${kb.name}"? This will permanently remove all documents and API keys.`)) return;
    setDeleting(true);
    try {
      await deleteKB(kbId);
      navigate('/');
    } catch {
      alert('Failed to delete knowledge base');
      setDeleting(false);
    }
  };

  if (loading) {
    return <div className="text-center py-16 text-gray-400">Loading…</div>;
  }
  if (!kb) return null;

  const TABS: { id: Tab; label: string }[] = [
    { id: 'documents', label: 'Documents' },
    { id: 'chat',      label: 'Chat' },
    { id: 'apikeys',   label: 'API Keys' },
    { id: 'settings',  label: 'Settings' },
  ];

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
        <Link to="/" className="hover:text-blue-600 transition-colors">Knowledge Bases</Link>
        <span>/</span>
        <span className="text-gray-800 font-medium">{kb.name}</span>
      </div>

      {/* Page header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{kb.name}</h1>
          {kb.description && <p className="text-sm text-gray-500 mt-1">{kb.description}</p>}
          <div className="flex items-center gap-3 mt-2 text-xs text-gray-400">
            <span className="font-mono bg-gray-100 px-2 py-0.5 rounded">{kb.rag_type}</span>
            <span>{kb.total_documents} docs</span>
            <span>&middot;</span>
            <span>{kb.total_chunks} chunks</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex gap-0">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                tab === t.id
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      {tab === 'documents' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <FileUpload
                onUploadComplete={loadDocuments}
                onUpload={(file) => uploadFileToKB(kbId!, file)}
              />
            </div>
          </div>
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <DocumentList documents={documents} onRefresh={loadDocuments} />
            </div>
          </div>
        </div>
      )}

      {tab === 'chat' && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden" style={{ height: 'calc(100vh - 260px)' }}>
          <ChatInterface kbId={kbId!} sessionId={sessionId} />
        </div>
      )}

      {tab === 'apikeys' && (
        <div className="max-w-2xl">
          <APIKeysPanel kbId={kbId!} />
        </div>
      )}

      {tab === 'settings' && (
        <div className="max-w-2xl space-y-6">
          {/* Settings form */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-4">
            <h3 className="font-semibold text-gray-900">General Settings</h3>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={settingsForm.name ?? ''}
                onChange={(e) => setSettingsForm({ ...settingsForm, name: e.target.value })}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea
                value={settingsForm.description ?? ''}
                onChange={(e) => setSettingsForm({ ...settingsForm, description: e.target.value })}
                rows={2}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">RAG Pipeline Type</label>
              <p className="text-xs text-amber-600 mb-2">Changing this evicts the active pipeline &mdash; next query will re-initialise.</p>
              <RAGTypeSelector
                value={settingsForm.rag_type ?? 'hybrid'}
                onChange={(v) => setSettingsForm({ ...settingsForm, rag_type: v as RagType })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { key: 'chunk_size', label: 'Chunk Size' },
                { key: 'chunk_overlap', label: 'Chunk Overlap' },
                { key: 'top_k', label: 'Top K Results' },
                { key: 'retriever_k', label: 'Retriever K' },
                { key: 'graph_hops', label: 'Graph Hops' },
              ].map(({ key, label }) => (
                <div key={key}>
                  <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
                  <input
                    type="number"
                    value={(settingsForm as Record<string, unknown>)[key] as number ?? 0}
                    onChange={(e) => setSettingsForm({ ...settingsForm, [key]: parseInt(e.target.value) || 0 })}
                    className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleSaveSettings}
                disabled={saving}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {saving ? 'Saving…' : 'Save Settings'}
              </button>
              {saveMsg && <span className="text-sm text-gray-500">{saveMsg}</span>}
            </div>
          </div>

          {/* Danger zone */}
          <div className="bg-white rounded-xl border border-red-200 p-5 shadow-sm">
            <h3 className="font-semibold text-red-700 mb-1">Danger Zone</h3>
            <p className="text-sm text-gray-500 mb-4">
              Permanently delete this knowledge base, all its documents, and all API keys. This cannot be undone.
            </p>
            <button
              onClick={handleDeleteKB}
              disabled={deleting}
              className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              {deleting ? 'Deleting…' : `Delete "${kb.name}"`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
