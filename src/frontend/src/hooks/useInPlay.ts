import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchInPlay, type InPlayMatch } from '@/lib/api';

/**
 * Same cadence as the live board, and for the same reason: the re-pricing is
 * derived from a snapshot the results service refreshes once a minute, so a
 * faster poll recomputes identical numbers.
 */
const POLL_MS = 60_000;

/** The key a dashboard row joins on. Canonical names on both sides. */
export function inPlayKey(league: string, home: string, away: string): string {
  return `${league}|${home}|${away}`;
}

export interface UseInPlayState {
  /** Priced matches, indexed by `inPlayKey`. Empty while nothing is live. */
  byMatch: Map<string, InPlayMatch>;
  loading: boolean;
  /**
   * Set when the board could not be fetched. A row simply shows no in-play
   * block in that case — this exists so the page can say why if it wants to,
   * rather than leaving "nothing is being played" as the only reading.
   */
  error: string | null;
}

/**
 * Subscribes to the in-play board for the leagues on screen.
 *
 * Returns a lookup rather than a list because every consumer is a match row
 * asking "is this one live?", and scanning the array per row is the same
 * question answered n times.
 */
export function useInPlay(leagues?: string[], enabled: boolean = true): UseInPlayState {
  const [matches, setMatches] = useState<InPlayMatch[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const alive = useRef(true);
  const key = leagues?.join(',') ?? '';

  const load = useCallback(async () => {
    try {
      const board = await fetchInPlay(key ? key.split(',') : undefined);
      if (!alive.current) return;
      setMatches(board.matches);
      setError(null);
    } catch (err) {
      if (!alive.current) return;
      // The previous prices stay on screen. A momentary failure should not
      // make a match that is being played look as though it has stopped.
      setError(err instanceof Error ? err.message : 'Ao vivo indisponível');
    } finally {
      if (alive.current) setLoading(false);
    }
  }, [key]);

  useEffect(() => {
    // Nothing can be in progress on a day that is not today, so a dashboard
    // showing another date polls for nothing sixty times an hour.
    if (!enabled) {
      setMatches([]);
      setError(null);
      setLoading(false);
      return;
    }
    alive.current = true;
    setLoading(true);
    void load();
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [load, enabled]);

  const byMatch = useMemo(
    () =>
      new Map(
        matches.map((match) => [
          inPlayKey(match.league, match.home_team, match.away_team),
          match,
        ]),
      ),
    [matches],
  );

  return { byMatch, loading, error };
}
