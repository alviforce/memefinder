import { useRef, useEffect } from 'react';
import MemeCard from './MemeCard';

const MODE_LABEL = {
  all:   '🌟 Гибрид',
  text:  '📝 Текст',
  image: '🧠 Смысл',
  ocr:   '📝 OCR',
  clip:  '🧠 CLIP',
};

export default function MasonryGrid({ results, mode, meta, onMemeClick, onLoadMore, hasMore, loadingMore }) {
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

  const totalLabel = meta?.fused_total
    ? `${results.length} из ${meta.fused_total}${hasMore ? '' : ''}`
    : `${results.length}${hasMore ? '+' : ''}`;
  const timing = meta?.timing_ms?.total_ms;
  const retrievers = meta?.retrievers || [];

  return (
    <>
      <div className="results-header">
        <span className="results-count">
          Найдено: {totalLabel}
          {timing != null && (
            <span className="results-timing"> · {Math.round(timing)} мс</span>
          )}
        </span>
        <span>
          Режим: {MODE_LABEL[mode] || mode}
          {retrievers.length > 0 && (
            <span className="results-retrievers"> · {retrievers.join(' + ')}</span>
          )}
        </span>
      </div>
      <div className="masonry-grid">
        {results.map((meme, index) => (
          <MemeCard
            key={meme.filename}
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
