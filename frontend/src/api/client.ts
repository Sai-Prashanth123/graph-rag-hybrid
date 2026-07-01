import type { Document, HealthStatus, KnowledgeBase, APIKeyFull, APIKeyListItem, StreamEvent, RagType } from '../types';

const BASE = '/api';

// ── Health ────────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ── Knowledge Base CRUD ───────────────────────────────────────────────────────

export async function listKBs(): Promise<KnowledgeBase[]> {
  const res = await fetch(`${BASE}/kb`);
  if (!res.ok) throw new Error('Failed to list knowledge bases');
  return res.json();
}

export async function createKB(data: {
  name: string;
  description?: string;
  rag_type: RagType;
  chunk_size?: number;
  chunk_overlap?: number;
  top_k?: number;
  retriever_k?: number;
  graph_hops?: number;
}): Promise<KnowledgeBase> {
  const res = await fetch(`${BASE}/kb`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || 'Failed to create knowledge base');
  }
  return res.json();
}

export async function getKB(kbId: string): Promise<KnowledgeBase> {
  const res = await fetch(`${BASE}/kb/${kbId}`);
  if (!res.ok) throw new Error('Knowledge base not found');
  return res.json();
}

export async function updateKB(kbId: string, data: Partial<{
  name: string;
  description: string;
  rag_type: RagType;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  retriever_k: number;
  graph_hops: number;
}>): Promise<KnowledgeBase> {
  const res = await fetch(`${BASE}/kb/${kbId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update knowledge base');
  return res.json();
}

export async function deleteKB(kbId: string): Promise<void> {
  const res = await fetch(`${BASE}/kb/${kbId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete knowledge base');
}

// ── KB Documents & History ────────────────────────────────────────────────────

export async function fetchKBDocuments(kbId: string): Promise<Document[]> {
  const res = await fetch(`${BASE}/kb/${kbId}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function fetchKBHistory(kbId: string, sessionId?: string, limit = 20) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (sessionId) params.set('session_id', sessionId);
  const res = await fetch(`${BASE}/kb/${kbId}/history?${params}`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

// ── KB Upload ─────────────────────────────────────────────────────────────────

export async function uploadFileToKB(kbId: string, file: File): Promise<{
  success: boolean;
  doc_name: string;
  num_chunks: number;
  message: string;
}> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/kb/${kbId}/upload/file`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || 'Upload failed');
  }
  return res.json();
}

export async function uploadTextToKB(kbId: string, text: string, name: string) {
  const res = await fetch(`${BASE}/kb/${kbId}/upload/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, name }),
  });
  if (!res.ok) throw new Error('Text upload failed');
  return res.json();
}

// ── KB Chat (streaming) ───────────────────────────────────────────────────────

export async function* streamKBChat(
  kbId: string,
  question: string,
  sessionId: string,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/kb/${kbId}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });

  if (!res.ok) {
    throw new Error(`Stream request failed: ${res.status}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      for (const line of part.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            yield JSON.parse(line.slice(6)) as StreamEvent;
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  }
}

// ── API Keys ──────────────────────────────────────────────────────────────────

export async function listAPIKeys(kbId: string): Promise<APIKeyListItem[]> {
  const res = await fetch(`${BASE}/kb/${kbId}/keys`);
  if (!res.ok) throw new Error('Failed to list API keys');
  return res.json();
}

export async function createAPIKey(kbId: string, name: string): Promise<APIKeyFull> {
  const res = await fetch(`${BASE}/kb/${kbId}/keys`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error('Failed to create API key');
  return res.json();
}

export async function revokeAPIKey(kbId: string, keyId: string): Promise<void> {
  const res = await fetch(`${BASE}/kb/${kbId}/keys/${keyId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to revoke API key');
}

// ── Legacy single-KB helpers (kept for backward compat) ──────────────────────

export async function fetchDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE}/documents`).catch(() => null);
  if (!res || !res.ok) return [];
  return res.json();
}

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/upload/file`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

export async function fetchHistory(sessionId?: string, limit = 20) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (sessionId) params.set('session_id', sessionId);
  const res = await fetch(`${BASE}/history?${params}`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function* streamChat(
  question: string,
  sessionId: string,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Stream request failed: ${res.status}`);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';
    for (const part of parts) {
      for (const line of part.split('\n')) {
        if (line.startsWith('data: ')) {
          try { yield JSON.parse(line.slice(6)) as StreamEvent; } catch { /* skip */ }
        }
      }
    }
  }
}
