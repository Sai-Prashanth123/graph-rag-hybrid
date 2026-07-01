export interface Source {
  content: string;
  metadata: {
    chunk_id: string;
    name: string;
    chunk_index: number;
    file_path?: string;
    file_size?: number;
    graph_relevance?: number;
    [key: string]: unknown;
  };
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  execution_time?: number;
  timestamp: Date;
  streaming?: boolean;
}

export interface Document {
  document_name: string;
  document_type: string;
  num_chunks: number;
  upload_timestamp: string;
  file_size: number;
}

export interface HealthStatus {
  status: string;
  active_kbs?: number;
  components: {
    groq_llm: boolean;
    chromadb: boolean;
    bm25: boolean;
    neo4j: boolean;
    mysql: boolean;
  };
}

export interface StreamEvent {
  type: 'token' | 'done' | 'error';
  token?: string;
  sources?: Source[];
  session_id?: string;
  execution_time?: number;
  message?: string;
}

// ── B2B SaaS types ───────────────────────────────────────────────────────────

export type RagType = 'vector' | 'bm25' | 'hybrid' | 'graph' | 'full_hybrid';

export interface KnowledgeBase {
  kb_id: string;
  name: string;
  description: string | null;
  rag_type: RagType;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  retriever_k: number;
  graph_hops: number;
  created_at: string;
  updated_at: string;
  total_documents: number;
  total_chunks: number;
}

export interface APIKeyFull {
  key_id: string;
  api_key: string;
  name: string;
  kb_id: string;
  created_at: string;
  is_active: boolean;
  total_requests: number;
}

export interface APIKeyListItem {
  key_id: string;
  api_key_preview: string;
  name: string;
  kb_id: string;
  created_at: string;
  last_used: string | null;
  total_requests: number;
  is_active: boolean;
}
