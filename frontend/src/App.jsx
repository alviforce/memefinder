import { useState, useEffect, useCallback } from 'react';
import SearchBar from './components/SearchBar';
import MasonryGrid from './components/MasonryGrid';
import MemeModal from './components/MemeModal';
import IndexingPanel from './components/IndexingPanel';
import EmptyState from './components/EmptyState';
import { useInfiniteSearch, fetchStats } from './hooks/useSearch';

export default function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('all');
  const [imageFile, setImageFile] = useState(null);
  const [selectedMeme, setSelectedMeme] = useState(null);
  const [stats, setStats] = useState({
    meme_count: 0,
    embedding_count: 0,
    text_embedding_count: 0,
  });

  const { results, loading, loadingMore, error, searched, hasMore, loadMore, meta } =
    useInfiniteSearch(query, mode, imageFile);

  // Load stats
  const loadStats = useCallback(async () => {
    try {
      const data = await fetchStats();
      setStats(data);
    } catch (err) {
      console.warn('Failed to load stats:', err);
    }
  }, []);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const handleIndexingComplete = useCallback(() => {
    loadStats();
  }, [loadStats]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-logo">MemeFinder</h1>
        <p className="app-subtitle">Гибридный поиск мемов: OCR + BGE-M3 + CLIP</p>
      </header>

      {/* Search */}
      <SearchBar
        query={query}
        onQueryChange={setQuery}
        mode={mode}
        onModeChange={setMode}
        imageFile={imageFile}
        onImageChange={setImageFile}
      />

      {/* Stats */}
      <div className="stats-bar">
        <div className="stat-item">
          🖼️ <span className="stat-value">{stats.meme_count}</span> мемов
        </div>
        <div className="stat-item">
          🧠 <span className="stat-value">{stats.embedding_count}</span> CLIP
        </div>
        <div className="stat-item">
          📚 <span className="stat-value">{stats.text_embedding_count ?? 0}</span> BGE
        </div>
      </div>

      {/* Indexing Panel */}
      <IndexingPanel onIndexingComplete={handleIndexingComplete} />

      {/* Loading indicator (initial search) */}
      {loading && (
        <div className="skeleton-grid">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="skeleton-card"
              style={{ height: `${150 + Math.random() * 150}px` }}
            />
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="empty-state">
          <div className="empty-icon">⚠️</div>
          <h2 className="empty-title">Ошибка</h2>
          <p className="empty-description">{error}</p>
        </div>
      )}

      {/* Results with infinite scroll */}
      {!loading && !error && results.length > 0 && (
        <MasonryGrid
          results={results}
          mode={mode}
          meta={meta}
          onMemeClick={setSelectedMeme}
          onLoadMore={loadMore}
          hasMore={hasMore}
          loadingMore={loadingMore}
        />
      )}

      {/* Empty State */}
      {!loading && !error && results.length === 0 && (
        <EmptyState searched={searched} query={query} mode={mode} />
      )}

      {/* Modal */}
      {selectedMeme && (
        <MemeModal
          meme={selectedMeme}
          onClose={() => setSelectedMeme(null)}
        />
      )}
    </div>
  );
}
