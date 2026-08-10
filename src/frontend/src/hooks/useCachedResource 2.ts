import { useEffect, useState } from 'react';

/**
 * Session-scoped caching for a read that never changes while the app is open.
 *
 * The catalogue reads — leagues, team names — are immutable for the life of a
 * session, so they need none of the invalidation machinery
 * `PredictionsContext` exists to provide. A module-level shared promise is
 * enough, and it means several components can ask for the same resource while
 * only one request goes out.
 *
 * A failed load clears the cache, so a remount retries instead of latching the
 * error for the rest of the session.
 */
export interface CachedResource<T> {
  load: () => Promise<T>;
  /** Drops the cache. For tests and for an explicit refresh. */
  invalidate: () => void;
}

export function createCachedResource<T>(fetcher: () => Promise<T>): CachedResource<T> {
  let inFlight: Promise<T> | null = null;

  return {
    load(): Promise<T> {
      if (!inFlight) {
        inFlight = fetcher().catch((error: unknown) => {
          inFlight = null;
          throw error;
        });
      }
      return inFlight;
    },
    invalidate(): void {
      inFlight = null;
    },
  };
}

export interface CachedResourceState<T> {
  /** Null until the first load resolves, and if it fails. */
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Subscribes a component to a cached resource.
 *
 * `resource` must be module-level so its identity is stable — a resource
 * created inside a component would rebuild its cache on every render, which
 * is the whole thing this exists to avoid.
 */
export function useCachedResource<T>(
  resource: CachedResource<T>,
  errorMessage: string,
): CachedResourceState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    // Resolves asynchronously, so nothing is set during the effect body.
    resource
      .load()
      .then((result) => {
        if (!active) return;
        setData(result);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : errorMessage);
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [resource, errorMessage]);

  return { data, loading, error };
}
