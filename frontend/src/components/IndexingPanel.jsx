import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchStats, startIndexing } from '../hooks/useSearch';

export default function IndexingPanel({ onIndexingComplete }) {
  const [status, setStatus] = useState({
    status: 'idle',
    total: 0,
    processed: 0,
    errors: 0,
    elapsed_seconds: 0,
  });
  const [starting, setStarting] = useState(false);
  const pollRef = useRef(null);

  const pollStatus = useCallback(async () => {
    try {
      const data = await fetchStats();
      const indexing = data.indexing || {};
      setStatus(indexing);

      if (indexing.status === 'done' || indexing.status === 'error') {
        clearInterval(pollRef.current);
        pollRef.current = null;
        if (indexing.status === 'done' && onIndexingComplete) {
          onIndexingComplete();
        }
      }
    } catch (err) {
      console.error('Failed to fetch index status:', err);
    }
  }, [onIndexingComplete]);

  // Initial status check
  useEffect(() => {
    pollStatus();
  }, [pollStatus]);

  // If status is running when component mounts, start polling
  useEffect(() => {
    if (status.status === 'running' && !pollRef.current) {
      pollRef.current = setInterval(pollStatus, 1500);
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [status.status, pollStatus]);

  const handleStart = async () => {
    setStarting(true);
    try {
      await startIndexing();
      // Start polling
      pollRef.current = setInterval(pollStatus, 1500);
      // Immediate check
      await pollStatus();
    } catch (err) {
      console.error('Failed to start indexing:', err);
      alert('Ошибка запуска индексации: ' + err.message);
    } finally {
      setStarting(false);
    }
  };

  const progress =
    status.total > 0 ? Math.round((status.processed / status.total) * 100) : 0;

  const isRunning = status.status === 'running';
  const isDone = status.status === 'done';

  const formatTime = (s) => {
    if (!s) return '0с';
    const mins = Math.floor(s / 60);
    const secs = Math.round(s % 60);
    return mins > 0 ? `${mins}м ${secs}с` : `${secs}с`;
  };

  return (
    <div className="indexing-panel">
      <div className="indexing-header">
        <div>
          <div className="indexing-title">
            {isRunning ? '⚙️ Индексация...' : isDone ? '✅ Индексация завершена' : '📂 Индексация мемов'}
          </div>
        </div>
      </div>

      {/* Start button */}
      {!isRunning && (
        <div className="indexing-input-row" style={{justifyContent: 'center'}}>
          <button
            id="start-indexing-btn"
            className="indexing-btn"
            onClick={handleStart}
            disabled={starting}
          >
            {starting ? 'Запуск...' : isDone ? '🔄 Индексировать новую партию' : '🚀 Начать индексацию из папки memes/'}
          </button>
        </div>
      )}

      {(isRunning || isDone) && (
        <div className="indexing-progress">
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="indexing-status">
            <span className="current-file">
              {isRunning
                ? `Обрабатывается...`
                : isDone
                ? `Обработано: ${status.processed} фото`
                : ''}
            </span>
            <span>
              {status.processed}/{status.total} · {formatTime(status.elapsed_seconds)}
              {status.errors > 0 && ` · ❌ ${status.errors} ошибок`}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
