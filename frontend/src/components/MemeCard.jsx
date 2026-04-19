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
      // Fallback: copy the proxy URL
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

  return (
    <div
      className="meme-card"
      onClick={onClick}
      style={{ animationDelay: `${Math.min(index * 50, 500)}ms` }}
    >
      {/* Score badge */}
      {meme.score != null && (
        <span className="meme-card-score">
          {Math.round(meme.score * 100)}%
        </span>
      )}

      {/* Thumbnail from base64 (zero network requests) */}
      <img
        className="meme-card-image"
        src={meme.thumbnail_base64}
        alt={meme.ocr_text || 'Мем'}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        style={{ opacity: loaded ? 1 : 0, transition: 'opacity 0.3s ease' }}
      />

      {/* Hover overlay */}
      <div className="meme-card-overlay">
        {mode === 'ocr' && meme.ocr_text && (
          <div className="meme-card-text">{meme.ocr_text}</div>
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
