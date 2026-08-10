import { useMemo } from 'react';
import { fetchLeagues } from '@/lib/api';
import type { League } from '@/types';
import { createCachedResource, useCachedResource } from './useCachedResource';

/** Module-level, so every caller shares one request and one cache. */
const leaguesResource = createCachedResource(fetchLeagues);

/** Stable empty reference, so `domestic` does not recompute on every render. */
const NO_LEAGUES: League[] = [];

/** Drops the cached league list. */
export const invalidateLeagues = leaguesResource.invalidate;

export interface UseLeaguesResult {
  /** Everything the backend offers, competitions included. */
  leagues: League[];
  /**
   * Domestic divisions only.
   *
   * What the team pickers must use: `predictCustom` and `fetchTeamStats` take
   * a code that has to resolve to a football-data division, and a UEFA
   * competition has no such feed.
   */
  domestic: League[];
  loading: boolean;
  error: string | null;
}

/** The selectable leagues, from the backend — never a compile-time list. */
export function useLeagues(): UseLeaguesResult {
  const { data, loading, error } = useCachedResource(
    leaguesResource,
    'Erro ao carregar ligas',
  );

  const leagues = data ?? NO_LEAGUES;
  const domestic = useMemo(
    () => leagues.filter((league) => league.type === 'league'),
    [leagues],
  );

  return { leagues, domestic, loading, error };
}
