import type { HealthStatus } from '../types';

interface Props {
  health: HealthStatus;
}

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-xs text-gray-500">
      <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${ok ? 'bg-green-500' : 'bg-red-400'}`} />
      {label}
    </span>
  );
}

export default function StatusBar({ health }: Props) {
  const c = health.components;
  return (
    <div>
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">System Status</p>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        <Dot ok={c.groq_llm} label="Groq LLM" />
        <Dot ok={c.chromadb} label="Vector DB" />
        <Dot ok={c.bm25} label="BM25" />
        <Dot ok={c.neo4j} label="Neo4j" />
        <Dot ok={c.mysql} label="MySQL" />
      </div>
    </div>
  );
}
