import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchLiveResults, type LiveResults } from '@/lib/api';

/**
 * Poll interval, matched to the results service's own TTL.
 *
 * Polling faster buys nothing: inside its window the service serves the same
 * cached snapshot without contacting a provider, so the extra requests would
 * return identical bytes. Polling slower would let the board lag behind a
 * refresh that has already happened.
 */
const POLL_MS = 60_000;

export interface UseLiveResultsState {
  data: LiveResults | null;
  loading: boolean;
  /**
   * Set when the results service could not answer. Kept separate from an
   * empty board: "nothing is being played" and "we could not find out" look
   * identical on screen and call for opposite messages.
   */
  error: string | null;
  refresh: () => void;
}

/**
 * Subscribes to the live board, refreshing on a timer while mounted.
 *
 * Deliberately not a `createCachedResource`: those exist for reads that never
 * change during a session, which is the opposite of this.
 */
export function useLiveResults(leagues?: string[]): UseLiveResultsState {
  const [data, setData] = useState<LiveResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Serialised so a slow response cannot be applied after a newer one, and so
  // an unmount mid-flight does not set state on a dead component.
  const alive = useRef(true);
  const key = leagues?.join(',') ?? '';

  const load = useCallback(async () => {
    try {
      const next = await fetchLiveResults(key ? key.split(',') : undefined);
      if (!alive.current) return;
      setData(next);
      setError(null);
    } catch (err) {
      if (!alive.current) return;
      // The previous board is left in place: last known scores beat a screen
      // that empties itself the moment the service hiccups.
      setError(err instanceof Error ? err.message : 'Resultados indisponíveis');
    } finally {
      if (alive.current) setLoading(false);
    }
  }, [key]);

  useEffect(() => {
    alive.current = true;
    setLoading(true);
    void load();
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [load]);

  return { data, loading, error, refresh: () => void load() };
}
