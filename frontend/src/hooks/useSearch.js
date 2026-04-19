import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = '/api';

/**
 * Hook for infinite-scroll meme search with pagination.
 * Loads PAGE_SIZE results at a time using offset.
 */
const PAGE_SIZE = 30;

export function useInfiniteSearch(query, mode, debounceMs = 400) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const abortRef = useRef(null);
  const offsetRef = useRef(0);

  const doSearch = useCallback(async (q, m, offset = 0, append = false) => {
    // Abort previous request
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
    }
    setError(null);

    try {
      const res = await fetch(
        `${API_BASE}/search?q=${encodeURIComponent(q)}&mode=${m}&limit=${PAGE_SIZE}&offset=${offset}`,
        { signal: controller.signal }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const items = data.results || [];

      if (append) {
        setResults(prev => [...prev, ...items]);
      } else {
        setResults(items);
      }

      setHasMore(items.length >= PAGE_SIZE);
      offsetRef.current = offset + items.length;
      setSearched(true);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
        console.error('Search failed:', err);
      }
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, []);

  // Debounced search — reset on query/mode change
  useEffect(() => {
    if (!query || query.trim().length === 0) {
      setResults([]);
      setSearched(false);
      setError(null);
      setHasMore(false);
      offsetRef.current = 0;
      return;
    }

    const timer = setTimeout(() => {
      offsetRef.current = 0;
      doSearch(query.trim(), mode, 0, false);
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [query, mode, debounceMs, doSearch]);

  // Load more (called by IntersectionObserver)
  const loadMore = useCallback(() => {
    if (!query || loading || loadingMore || !hasMore) return;
    doSearch(query.trim(), mode, offsetRef.current, true);
  }, [query, mode, loading, loadingMore, hasMore, doSearch]);

  return { results, loading, loadingMore, error, searched, hasMore, loadMore };
}

/**
 * Fetch stats + indexing progress.
 */
export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Start indexing the local 'memes' folder.
 */
export async function startIndexing() {
  const res = await fetch(`${API_BASE}/index/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Build the original image URL for a meme (from local memes folder).
 */
export function originalImageUrl(filename) {
  return `${API_BASE}/image/original/${encodeURIComponent(filename)}`;
}

/**
 * Copy an image to clipboard via Canvas (cross-browser compatible).
 * Fetches the original image, draws on canvas, converts to PNG.
 */
export async function copyImageToClipboard(filename) {
  const url = originalImageUrl(filename);
  const response = await fetch(url);
  const blob = await response.blob();

  // Create an image element
  const img = document.createElement('img');
  img.crossOrigin = 'anonymous';

  return new Promise((resolve, reject) => {
    img.onload = async () => {
      try {
        // Draw on canvas
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        // Convert to PNG blob
        const pngBlob = await new Promise(res =>
          canvas.toBlob(res, 'image/png')
        );

        // Write to clipboard
        await navigator.clipboard.write([
          new ClipboardItem({ 'image/png': pngBlob }),
        ]);

        URL.revokeObjectURL(img.src);
        resolve(true);
      } catch (err) {
        URL.revokeObjectURL(img.src);
        reject(err);
      }
    };

    img.onerror = (err) => {
      URL.revokeObjectURL(img.src);
      reject(err);
    };

    img.src = URL.createObjectURL(blob);
  });
}
