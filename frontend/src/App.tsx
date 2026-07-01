import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import KBListPage from './pages/KBListPage';
import KBDetailPage from './pages/KBDetailPage';

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      {/* Offset for fixed navbar (56px = h-14) */}
      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<KBListPage />} />
            <Route path="/kb/:kbId" element={<KBDetailPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
