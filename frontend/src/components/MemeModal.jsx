import { useState, useEffect, useCallback } from 'react';
import { originalImageUrl, copyImageToClipboard } from '../hooks/useSearch';

export default function MemeModal({ meme, onClose }) {
  const [copied, setCopied] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);

  // Close on Escape
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handler);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) onClose();
  };

  const handleCopy = useCallback(async () => {
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

  const handleDownload = useCallback(async () => {
    try {
      const url = originalImageUrl(meme.filename);
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = meme.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Download failed:', err);
    }
  }, [meme.filename]);

  if (!meme) return null;

  const imgSrc = originalImageUrl(meme.filename);

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        <button className="modal-close" onClick={onClose} aria-label="Закрыть">
          ✕
        </button>

        {/* Loading spinner while original loads */}
        {!imgLoaded && (
          <div className="modal-image-loader">
            <div className="spinner" />
            <span>Загрузка оригинала...</span>
          </div>
        )}

        <img
          className="modal-image"
          src={imgSrc}
          alt={meme.ocr_text || 'Мем'}
          onLoad={() => setImgLoaded(true)}
          style={{ display: imgLoaded ? 'block' : 'none' }}
        />

        {meme.ocr_text && (
          <div className="modal-ocr-text">
            📝 {meme.ocr_text}
          </div>
        )}

        <div className="modal-toolbar">
          <button
            className={`modal-btn primary ${copied ? 'copied' : ''}`}
            onClick={handleCopy}
          >
            {copied ? '✓ Скопировано!' : '📋 Скопировать'}
          </button>
          <button className="modal-btn" onClick={handleDownload}>
            ⬇️ Скачать
          </button>
          <button className="modal-btn" onClick={onClose}>
            ✕ Закрыть
          </button>
        </div>
      </div>
    </div>
  );
}
