import { useState, useCallback } from 'react';
import { originalImageUrl, copyImageToClipboard } from '../hooks/useSearch';

export default function MemeCard({ meme, mode, index, onClick }) {
  const [copied, setCopied] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const handleCopy = useCallback(async (e) => {
    e.stopPropagation();
    try {
      await copyImageToClipboard(meme.filename);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
      try {
        const url = window.location.origin + originalImageUrl(meme.filename);
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (e2) {
        console.error('Fallback copy failed:', e2);
      }
    }
  }, [meme.filename]);

  const tags = Array.isArray(meme.tags) ? meme.tags : [];
  const caption = (meme.caption || '').trim();

  return (
    <div
      className="meme-card"
      onClick={onClick}
      style={{ animationDelay: `${Math.min(index * 50, 500)}ms` }}
    >
      {/* Score badge */}
      {meme.score != null && (
        <span className="meme-card-score">
          {meme.score < 1 ? Math.round(meme.score * 100) + '%' : meme.score.toFixed(2)}
        </span>
      )}

      {/* Thumbnail from base64 (zero network requests) */}
      <img
        className="meme-card-image"
        src={meme.thumbnail_base64}
        alt={caption || meme.ocr_text || 'Мем'}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        style={{ opacity: loaded ? 1 : 0, transition: 'opacity 0.3s ease' }}
      />

      {/* Hover overlay */}
      <div className="meme-card-overlay">
        {caption && (
          <div className="meme-card-caption" title={caption}>
            💬 {caption}
          </div>
        )}
        {!caption && mode === 'text' && meme.ocr_text && (
          <div className="meme-card-text">{meme.ocr_text}</div>
        )}
        {tags.length > 0 && (
          <div className="meme-card-tags">
            {tags.slice(0, 5).map((t) => (
              <span key={t} className="meme-tag">#{t}</span>
            ))}
          </div>
        )}
        <div className="meme-card-actions">
          <button
            className={`meme-action-btn ${copied ? 'copied' : ''}`}
            onClick={handleCopy}
          >
            {copied ? '✓ Скопировано' : '📋 Копировать'}
          </button>
        </div>
      </div>
    </div>
  );
}
