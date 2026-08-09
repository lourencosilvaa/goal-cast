/**
 * Pure derivations behind the dashboard.
 *
 * Kept apart from the components so those files export components only —
 * mixing the two breaks React Fast Refresh (and the lint rule guarding it),
 * the same split `stats/format.ts` makes.
 *
 * Every figure here comes out of the prediction payload the list already
 * renders. There is no metrics endpoint, so nothing on the dashboard is a
 * number the backend could not reproduce.
 */
import { ticker } from '@/config/theme';
import type { LeaguePredictions, MatchPrediction } from '@/types';

export interface HeroMatch {
  match: MatchPrediction;
  leagueCode: string;
}

export interface DashboardMetrics {
  matchCount: number;
  leagueCount: number;
  valueBetCount: number;
  /** Best edge across every value bet, as a percentage, or null when none. */
  bestEdge: number | null;
  /** One entry per league with fixtures, for the distribution strip. */
  perLeague: { code: string; name: string; count: number }[];
}

/**
 * Builds the ticker's contents.
 *
 * Only high-confidence non-draw calls make it in: a scrolling line of
 * coin-flips would be noise, and the point of the strip is to surface the
 * handful of fixtures the model is actually opinionated about.
 */
export function buildSignals(leagues: LeaguePredictions[]): string[] {
  return leagues
    .flatMap((league) => league.matches)
    .filter((match) => match.confidence >= ticker.minConfidence)
    .filter((match) => match.predicted_outcome !== 'Draw')
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, ticker.maxItems)
    .map((match) => {
      const pick = match.predicted_outcome === 'Home Win' ? match.home_team : match.away_team;
      const against = match.predicted_outcome === 'Home Win' ? match.away_team : match.home_team;
      return `${(match.confidence * 100).toFixed(0)}% — ${pick} vence ${against}`;
    });
}

/** The day's most confident call, or null when nothing is scheduled. */
export function pickHeroMatch(leagues: LeaguePredictions[]): HeroMatch | null {
  let best: HeroMatch | null = null;
  for (const league of leagues) {
    for (const match of league.matches) {
      if (!best || match.confidence > best.match.confidence) {
        best = { match, leagueCode: league.league_code };
      }
    }
  }
  return best;
}

/** Derives every rail figure from the filtered payload in a single pass. */
export function deriveMetrics(leagues: LeaguePredictions[]): DashboardMetrics {
  const perLeague = leagues
    .map((league) => ({
      code: league.league_code,
      name: league.league_name,
      count: league.matches.length,
    }))
    .filter((entry) => entry.count > 0);

  const edges = leagues.flatMap((league) =>
    league.matches.flatMap((match) => match.value_bets.map((bet) => bet.edge)),
  );

  return {
    matchCount: perLeague.reduce((total, entry) => total + entry.count, 0),
    leagueCount: perLeague.length,
    valueBetCount: edges.length,
    bestEdge: edges.length > 0 ? Math.max(...edges) : null,
    perLeague,
  };
}

/** 1X2 shorthand for a predicted outcome. */
export function outcomeShorthand(match: MatchPrediction): string {
  if (match.predicted_outcome === 'Home Win') return '1';
  if (match.predicted_outcome === 'Away Win') return '2';
  return 'X';
}
