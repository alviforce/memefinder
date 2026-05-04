import { useRef, useEffect } from 'react';

const MODES = [
  { id: 'all',  icon: '🌟', label: 'Гибридный',  description: 'OCR + смысл + картинка' },
  { id: 'text', icon: '📝', label: 'По тексту',  description: 'OCR FTS + BGE-M3' },
  { id: 'image',icon: '🧠', label: 'По смыслу',  description: 'CLIP визуал' },
];

export default function SearchBar({
  query,
  onQueryChange,
  mode,
  onModeChange,
  imageFile,
  onImageChange,
}) {
  const inputRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Keyboard shortcut: Ctrl+K or / to focus
  useEffect(() => {
    const handler = (e) => {
      if (
        (e.ctrlKey && e.key === 'k') ||
        (e.key === '/' && document.activeElement !== inputRef.current)
      ) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const placeholderByMode = {
    all: 'Что ищем? Опишите мем или вставьте текст с картинки…',
    text: 'Поиск по тексту на мемах…',
    image: 'Опишите мем словами или загрузите картинку →',
  };

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (f) onImageChange?.(f);
    // Reset so picking the same file twice still triggers change
    e.target.value = '';
  };

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
            placeholder={placeholderByMode[mode] || placeholderByMode.all}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            autoComplete="off"
            spellCheck="false"
          />
          {/* Image upload */}
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFile}
          />
          <button
            type="button"
            className={`search-image-btn ${imageFile ? 'active' : ''}`}
            onClick={() => fileRef.current?.click()}
            title={imageFile ? `Поиск по: ${imageFile.name}` : 'Поиск по картинке'}
          >
            {imageFile ? '🖼️' : '📷'}
          </button>
          {imageFile && (
            <button
              type="button"
              className="search-image-clear"
              onClick={() => onImageChange?.(null)}
              title="Убрать картинку"
            >
              ✕
            </button>
          )}
        </div>

        {/* Mode Toggle */}
        <div className="mode-toggle" role="tablist">
          {MODES.map(({ id, icon, label, description }) => (
            <button
              key={id}
              id={`mode-${id}`}
              className={`mode-toggle-btn ${mode === id ? 'active' : ''}`}
              onClick={() => onModeChange(id)}
              role="tab"
              aria-selected={mode === id}
            >
              <span className="mode-label">{icon} {label}</span>
              <span className="mode-description">{description}</span>
            </button>
          ))}
        </div>

        {imageFile && (
          <div className="search-image-preview">
            <span>🖼️ Картинка: <strong>{imageFile.name}</strong> · поиск по схожести</span>
          </div>
        )}
      </div>
    </div>
  );
}
