import { useState, useRef, useEffect } from 'react';

export default function SearchBar({ query, onQueryChange, mode, onModeChange }) {
  const inputRef = useRef(null);

  // Focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Keyboard shortcut: Ctrl+K or / to focus
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && document.activeElement !== inputRef.current)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <div className="search-container">
      <div className="search-wrapper">
        {/* Search Input */}
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            ref={inputRef}
            id="search-input"
            className="search-input"
            type="text"
            placeholder={mode === 'ocr'
              ? 'Поиск по тексту на мемах...'
              : 'Опишите мем, который ищете...'
            }
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            autoComplete="off"
            spellCheck="false"
          />
        </div>

        {/* Mode Toggle */}
        <div className="mode-toggle" role="tablist">
          <button
            id="mode-clip"
            className={`mode-toggle-btn ${mode === 'clip' ? 'active' : ''}`}
            onClick={() => onModeChange('clip')}
            role="tab"
            aria-selected={mode === 'clip'}
          >
            <span className="mode-label">🧠 По смыслу (CLIP)</span>
            <span className="mode-description">Семантический поиск</span>
          </button>
          <button
            id="mode-ocr"
            className={`mode-toggle-btn ${mode === 'ocr' ? 'active' : ''}`}
            onClick={() => onModeChange('ocr')}
            role="tab"
            aria-selected={mode === 'ocr'}
          >
            <span className="mode-label">📝 По тексту (OCR)</span>
            <span className="mode-description">Надписи на картинках</span>
          </button>
        </div>
      </div>
    </div>
  );
}
