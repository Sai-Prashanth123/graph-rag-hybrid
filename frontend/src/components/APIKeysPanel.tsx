import { useEffect, useState } from 'react';
import { listAPIKeys, createAPIKey, revokeAPIKey } from '../api/client';
import type { APIKeyFull, APIKeyListItem } from '../types';

interface Props {
  kbId: string;
}

export default function APIKeysPanel({ kbId }: Props) {
  const [keys, setKeys] = useState<APIKeyListItem[]>([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [generating, setGenerating] = useState(false);
  const [revealedKey, setRevealedKey] = useState<APIKeyFull | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');

  const loadKeys = async () => {
    try {
      setKeys(await listAPIKeys(kbId));
    } catch {
      /* silent */
    }
  };

  useEffect(() => {
    loadKeys();
  }, [kbId]);

  const handleGenerate = async () => {
    if (!newKeyName.trim()) return;
    setGenerating(true);
    setError('');
    try {
      const full = await createAPIKey(kbId, newKeyName.trim());
      setRevealedKey(full);
      setNewKeyName('');
      loadKeys();
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  };

  const handleRevoke = async (keyId: string) => {
    if (!confirm('Revoke this key? Apps using it will lose access immediately.')) return;
    try {
      await revokeAPIKey(kbId, keyId);
      loadKeys();
    } catch (e) {
      setError(String(e));
    }
  };

  const copyKey = () => {
    if (!revealedKey) return;
    navigator.clipboard.writeText(revealedKey.api_key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* One-time key reveal */}
      {revealedKey && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-amber-800 mb-2">
            Store this key now &mdash; it won't be shown again
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 bg-white border border-amber-200 rounded-lg px-3 py-2 text-sm font-mono text-gray-800 break-all">
              {revealedKey.api_key}
            </code>
            <button
              onClick={copyKey}
              className="flex-shrink-0 px-3 py-2 bg-amber-600 text-white text-sm rounded-lg hover:bg-amber-700 transition-colors"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <button
            onClick={() => setRevealedKey(null)}
            className="mt-3 text-xs text-amber-700 hover:underline"
          >
            I've saved it &mdash; dismiss
          </button>
        </div>
      )}

      {/* Generate new key */}
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <h3 className="font-semibold text-gray-900 mb-3">Generate New API Key</h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            placeholder="Key name (e.g. My App)"
            className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          />
          <button
            onClick={handleGenerate}
            disabled={generating || !newKeyName.trim()}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {generating ? 'Generating…' : 'Generate'}
          </button>
        </div>
        {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
      </div>

      {/* Key list */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-100">
          <h3 className="font-semibold text-gray-900 text-sm">Active Keys ({keys.length})</h3>
        </div>

        {keys.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">No API keys yet.</p>
        ) : (
          <ul className="divide-y divide-gray-100">
            {keys.map((k) => (
              <li key={k.key_id} className="px-4 py-3 flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900 text-sm">{k.name}</span>
                    {!k.is_active && (
                      <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">revoked</span>
                    )}
                  </div>
                  <code className="text-xs text-gray-400 font-mono">{k.api_key_preview}</code>
                  <div className="text-xs text-gray-400 mt-0.5">
                    Created {new Date(k.created_at).toLocaleDateString()}
                    {k.last_used && ` · Last used ${new Date(k.last_used).toLocaleDateString()}`}
                    {' · '}{k.total_requests} requests
                  </div>
                </div>
                {k.is_active && (
                  <button
                    onClick={() => handleRevoke(k.key_id)}
                    className="flex-shrink-0 text-xs text-red-500 hover:text-red-700 transition-colors"
                  >
                    Revoke
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Usage example */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <h3 className="font-semibold text-gray-700 text-sm mb-2">Usage Example</h3>
        <pre className="text-xs text-gray-600 overflow-x-auto whitespace-pre-wrap">
{`curl -X POST https://your-domain.com/v1/chat \\
  -H "Authorization: Bearer rag-xxxxxxxx-..." \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is the refund policy?"}'`}
        </pre>
      </div>
    </div>
  );
}
