import { useCallback } from 'react';
import { fetchTeams } from '@/lib/api';
import { createCachedResource, useCachedResource } from './useCachedResource';

type TeamsByLeague = Record<string, string[]>;

/** Module-level, so every caller shares one request and one cache. */
const teamsResource = createCachedResource(fetchTeams);

/** Stable empty reference, so consumers do not see a new object per render. */
const NO_TEAMS: TeamsByLeague = {};

/** Drops the cached team catalogue. */
export const invalidateTeams = teamsResource.invalidate;

export interface UseTeamsResult {
  /** Canonical team names keyed by league code. */
  teamsByLeague: TeamsByLeague;
  /** Names for one league, or an empty list while loading or if absent. */
  teamsFor: (leagueCode: string) => string[];
  loading: boolean;
  error: string | null;
}

/**
 * The team catalogue, from the backend.
 *
 * The backend scopes this to `data.served_leagues`, so a league withdrawn
 * from the product contributes no names here either — the pickers cannot
 * offer a team from a division nobody can select.
 *
 * The error is surfaced rather than swallowed: the previous per-page loads
 * ended in `.catch(() => {})`, which turned a failed catalogue into a silently
 * empty picker with nothing to explain it.
 */
export function useTeams(): UseTeamsResult {
  const { data, loading, error } = useCachedResource(
    teamsResource,
    'Erro ao carregar equipas',
  );

  const teamsByLeague = data ?? NO_TEAMS;
  const teamsFor = useCallback(
    (leagueCode: string): string[] => teamsByLeague[leagueCode] ?? [],
    [teamsByLeague],
  );

  return { teamsByLeague, teamsFor, loading, error };
}
