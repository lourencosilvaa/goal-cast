import type { LeaguePredictions, MatchPrediction } from '@/types';

/** URL-safe slug for a team name. */
function slug(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

/** Stable, deep-linkable identifier for a match within a league. */
export function buildMatchId(leagueCode: string, home: string, away: string): string {
  return `${leagueCode}__${slug(home)}__${slug(away)}`;
}

export interface ResolvedMatch {
  leagueCode: string;
  leagueName: string;
  match: MatchPrediction;
}

/** Finds a match across all leagues by its `buildMatchId` identifier. */
export function findMatchById(
  leagues: LeaguePredictions[],
  matchId: string,
): ResolvedMatch | null {
  for (const league of leagues) {
    for (const match of league.matches) {
      if (buildMatchId(league.league_code, match.home_team, match.away_team) === matchId) {
        return {
          leagueCode: league.league_code,
          leagueName: league.league_name,
          match,
        };
      }
    }
  }
  return null;
}
