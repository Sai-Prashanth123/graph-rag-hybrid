import { useRef, useState } from 'react';

interface Props {
  onUploadComplete: () => void;
  onUpload: (file: File) => Promise<{ doc_name: string; num_chunks: number; message: string }>;
}

type Status =
  | { type: 'idle' }
  | { type: 'uploading'; name: string }
  | { type: 'success'; message: string }
  | { type: 'error'; message: string };

export default function FileUpload({ onUploadComplete, onUpload }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<Status>({ type: 'idle' });
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      setStatus({ type: 'uploading', name: file.name });
      try {
        const res = await onUpload(file);
        setStatus({ type: 'success', message: `${res.doc_name}: ${res.num_chunks} chunks indexed` });
        onUploadComplete();
      } catch (err) {
        setStatus({ type: 'error', message: `${file.name}: ${String(err)}` });
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    processFiles(e.dataTransfer.files);
  };

  return (
    <div>
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
        Upload Documents
      </h2>

      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer border-2 border-dashed rounded-xl p-5 text-center transition-colors select-none ${
          isDragging
            ? 'border-blue-500 bg-blue-50 text-blue-600'
            : 'border-gray-300 hover:border-blue-400 text-gray-400 hover:text-gray-500'
        }`}
      >
        <div className="text-2xl mb-1">↑</div>
        <p className="text-sm">Drop files or click to browse</p>
        <p className="text-xs mt-1 text-gray-400">PDF · TXT · MD</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown"
        multiple
        className="hidden"
        onChange={(e) => processFiles(e.target.files)}
      />

      {status.type === 'uploading' && (
        <div className="mt-2">
          <div className="h-1 rounded-full bg-gray-200 overflow-hidden">
            <div className="h-full bg-blue-500 animate-pulse w-full rounded-full" />
          </div>
          <p className="text-xs text-gray-500 mt-1 truncate">Uploading {status.name}…</p>
        </div>
      )}
      {status.type === 'success' && (
        <p className="text-xs text-green-600 mt-2 break-words">{status.message}</p>
      )}
      {status.type === 'error' && (
        <p className="text-xs text-red-500 mt-2 break-words">{status.message}</p>
      )}
    </div>
  );
}
