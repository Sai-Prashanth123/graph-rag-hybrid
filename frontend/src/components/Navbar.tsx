import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm h-14 flex items-center px-6">
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-3 hover:opacity-80 transition-opacity"
      >
        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
          R
        </div>
        <span className="font-semibold text-gray-900 text-base">RAG Platform</span>
      </button>

      <div className="ml-auto flex items-center gap-4 text-sm text-gray-500">
        <span>Powered by Groq · BAAI/bge-large</span>
      </div>
    </nav>
  );
}
