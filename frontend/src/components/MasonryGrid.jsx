import { useRef, useEffect, useCallback } from 'react';
import MemeCard from './MemeCard';

export default function MasonryGrid({ results, mode, onMemeClick, onLoadMore, hasMore, loadingMore }) {
  const sentinelRef = useRef(null);

  // IntersectionObserver for infinite scroll
  useEffect(() => {
    if (!sentinelRef.current || !onLoadMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loadingMore) {
          onLoadMore();
        }
      },
      { rootMargin: '200px' }
    );

    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [onLoadMore, hasMore, loadingMore]);

  if (!results || results.length === 0) return null;

  return (
    <>
      <div className="results-header">
        <span className="results-count">
          Найдено: {results.length} мем{results.length === 1 ? '' : results.length < 5 ? 'а' : 'ов'}
          {hasMore && '+'}
        </span>
        <span>
          Режим: {mode === 'clip' ? '🧠 CLIP' : '📝 OCR'}
        </span>
      </div>
      <div className="masonry-grid">
        {results.map((meme, index) => (
          <MemeCard
            key={`${meme.chat_id}_${meme.message_id}`}
            meme={meme}
            mode={mode}
            index={index}
            onClick={() => onMemeClick(meme)}
          />
        ))}
      </div>

      {/* Infinite scroll sentinel */}
      {hasMore && (
        <div ref={sentinelRef} className="load-more-sentinel">
          {loadingMore && (
            <div className="load-more-spinner">
              <div className="spinner" />
              <span>Загрузка...</span>
            </div>
          )}
        </div>
      )}
    </>
  );
}
