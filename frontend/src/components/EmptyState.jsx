const MODE_NAME = {
  all: 'Гибридном',
  text: 'Текстовом',
  image: 'Смысловом',
  ocr: 'OCR',
  clip: 'CLIP',
};

export default function EmptyState({ searched, query, mode }) {
  // No search yet
  if (!searched) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔎</div>
        <h2 className="empty-title">Найди свой мем</h2>
        <p className="empty-description">
          Введите запрос или загрузите картинку.
          <br />
          Режим <strong>🌟 Гибрид</strong> объединяет OCR, BGE-M3 и CLIP через RRF —
          лучший результат «из коробки».
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
        По запросу «{query}» в {MODE_NAME[mode] || mode} режиме ничего не нашлось.
        Попробуйте переформулировать запрос или переключить режим.
      </p>
    </div>
  );
}
