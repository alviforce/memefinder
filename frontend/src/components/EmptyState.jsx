export default function EmptyState({ searched, query, mode }) {
  // No search yet
  if (!searched) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔎</div>
        <h2 className="empty-title">Найди свой мем</h2>
        <p className="empty-description">
          Введите запрос в строку поиска.
          Используйте режим <strong>CLIP</strong> для поиска по смыслу
          или <strong>OCR</strong> для поиска по надписям.
        </p>
      </div>
    );
  }

  // Search returned no results
  return (
    <div className="empty-state">
      <div className="empty-icon">😕</div>
      <h2 className="empty-title">Ничего не найдено</h2>
      <p className="empty-description">
        По запросу «{query}» в режиме {mode === 'clip' ? 'CLIP' : 'OCR'} ничего не нашлось.
        Попробуйте переформулировать запрос или переключить режим поиска.
      </p>
    </div>
  );
}
