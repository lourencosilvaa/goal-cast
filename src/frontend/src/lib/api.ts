import { config } from '@/config';
import { devAuth } from '@/config';
import { supabase } from '@/lib/supabase';
import type { League, LeaguePredictions } from '@/types';

const BASE = config.apiUrl;

async function authHeaders(): Promise<Record<string, string>> {
  if (devAuth.enabled) {
    return { Authorization: `Bearer ${devAuth.token}` };
  }
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function fetchAvailableDates(): Promise<string[]> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/dates`, { headers });
  if (!res.ok) throw new Error('Failed to fetch dates');
  return res.json();
}

export async function fetchLeagues(): Promise<League[]> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/leagues`, { headers });
  if (!res.ok) throw new Error('Failed to fetch leagues');
  return res.json();
}

export async function fetchAllPredictions(date?: string): Promise<LeaguePredictions[]> {
  const headers = await authHeaders();
  const params = date ? `?date=${encodeURIComponent(date)}` : '';
  const res = await fetch(`${BASE}/api/predictions${params}`, { headers });
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function fetchLeaguePredictions(
  leagueCode: string,
  date?: string,
): Promise<LeaguePredictions> {
  const headers = await authHeaders();
  const params = date ? `?date=${encodeURIComponent(date)}` : '';
  const res = await fetch(`${BASE}/api/predictions/${leagueCode}${params}`, { headers });
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function refreshPredictions(leagueCode?: string): Promise<void> {
  const headers = await authHeaders();
  const params = leagueCode ? `?league_code=${leagueCode}` : '';
  await fetch(`${BASE}/api/predictions/refresh${params}`, { method: 'POST', headers });
}

export interface LiveMatch {
  league: string;
  kickoff: string;
  home_team: string;
  away_team: string;
  /** scheduled | live | paused | finished | postponed | cancelled */
  status: string;
  home_goals: number | null;
  away_goals: number | null;
  /** The provider's own clock, empty when it sends none. */
  minute: string;
  /** Minutes played, 0-90. Always present; see `minute_estimated`. */
  elapsed_minutes: number;
  /** True when the minute was derived from kick-off rather than observed. */
  minute_estimated: boolean;
  source: string;
  source_id: string;
}

export interface LiveEvent {
  /** goal | kickoff | full_time */
  type: string;
  match: LiveMatch;
  detected_at: string;
}

export interface LiveResults {
  fetched_at: string;
  source: string;
  /** The board is older than the service's staleness window. */
  stale: boolean;
  matches: LiveMatch[];
  events: LiveEvent[];
}

/**
 * Today's scores, proxied by the backend from the dedicated results service.
 *
 * A failure is thrown rather than flattened to an empty list: "nothing is
 * being played" and "we could not find out" render as the same blank board,
 * and only one of them is true.
 */
export async function fetchLiveResults(leagues?: string[]): Promise<LiveResults> {
  const headers = await authHeaders();
  const params = new URLSearchParams();
  if (leagues?.length) params.set('leagues', leagues.join(','));
  const res = await fetch(`${BASE}/api/results/live?${params}`, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Resultados indisponíveis' }));
    throw new Error(err.detail || 'Resultados indisponíveis');
  }
  return (await res.json()) as LiveResults;
}

export interface ProbabilityTriple {
  home_win: number;
  draw: number;
  away_win: number;
}

/** A match in progress, re-priced on its current score and time remaining. */
export interface InPlayMatch {
  league: string;
  /** Canonical spelling — the same key the prediction rows use. */
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
  elapsed_minutes: number;
  /** True when the minute was derived from kick-off rather than observed. */
  minute_estimated: boolean;
  status: string;
  /** What the model said before kick-off, kept for the comparison. */
  pre_match: ProbabilityTriple;
  /** The same model conditioned on the score and the minutes left. */
  live: ProbabilityTriple;
  remaining_minutes: number;
  /** Goals scored plus goals still expected, per side. */
  expected_home_goals: number;
  expected_away_goals: number;
}

/** A live match with no in-play number, and the reason there is none. */
export interface UnpricedMatch {
  league: string;
  home_team: string;
  away_team: string;
  /** unknown_team | no_prediction | no_expected_goals */
  reason: string;
}

export interface InPlayBoard {
  fetched_at: string;
  stale: boolean;
  matches: InPlayMatch[];
  unpriced: UnpricedMatch[];
}

/**
 * Live re-pricing for every match in progress that could be identified.
 *
 * Throws on failure for the same reason `fetchLiveResults` does: an empty
 * board is a real answer, and an unreachable service must not imitate it.
 */
export async function fetchInPlay(leagues?: string[]): Promise<InPlayBoard> {
  const headers = await authHeaders();
  const params = new URLSearchParams();
  if (leagues?.length) params.set('leagues', leagues.join(','));
  const res = await fetch(`${BASE}/api/in-play?${params}`, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Ao vivo indisponível' }));
    throw new Error(err.detail || 'Ao vivo indisponível');
  }
  return (await res.json()) as InPlayBoard;
}

/** The analysis back-ends the backend knows how to reach. */
export type AiProvider = 'gemini' | 'nvidia';

export interface AiAnalysis {
  analysis: string;
  /** Which back-end actually answered, for labelling the result. */
  provider: string;
}

export async function analyzeMatch(
  matchData: Record<string, unknown>,
  options: { provider?: AiProvider; model?: string } = {},
): Promise<AiAnalysis> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/ai/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({
      match_data: matchData,
      language: 'pt-PT',
      // Omitted rather than defaulted here: an empty model or provider means
      // "use what the server is configured for", and duplicating that default
      // in the client is how the two drift apart.
      ...(options.model && { model: options.model }),
      ...(options.provider && { provider: options.provider }),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'AI analysis failed' }));
    throw new Error(err.detail || 'AI analysis failed');
  }
  return (await res.json()) as AiAnalysis;
}

export async function getGeminiKeyStatus(): Promise<boolean> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/keys/gemini`, { headers });
  if (!res.ok) return false;
  const data = await res.json();
  return data.has_key as boolean;
}

export async function saveGeminiKey(key: string): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/keys/gemini`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ key }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save key' }));
    throw new Error(err.detail || 'Failed to save key');
  }
}

export async function deleteGeminiKey(): Promise<void> {
  const headers = await authHeaders();
  await fetch(`${BASE}/api/keys/gemini`, { method: 'DELETE', headers });
}

export async function getNvidiaKeyStatus(): Promise<boolean> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/keys/nvidia`, { headers });
  if (!res.ok) return false;
  const data = await res.json();
  return data.has_key as boolean;
}

export async function saveNvidiaKey(key: string): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/keys/nvidia`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ key }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save key' }));
    throw new Error(err.detail || 'Failed to save key');
  }
}

export async function deleteNvidiaKey(): Promise<void> {
  const headers = await authHeaders();
  await fetch(`${BASE}/api/keys/nvidia`, { method: 'DELETE', headers });
}

export interface InferencePrediction {
  home_team: string;
  away_team: string;
  predicted_outcome: string;
  confidence: number;
  probabilities?: { home_win: number; draw: number; away_win: number };
  league?: string;
  match_date?: string;
  time?: string;
}

export async function runInference(
  date?: string,
  leagueCode?: string,
): Promise<InferencePrediction[]> {
  const headers = await authHeaders();
  const params = new URLSearchParams();
  if (date) params.set('date', date);
  if (leagueCode) params.set('league_code', leagueCode);
  const res = await fetch(`${BASE}/api/predictions/infer?${params}`, {
    method: 'POST',
    headers,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Inference failed' }));
    throw new Error(err.detail || 'Inference failed');
  }
  const data = await res.json();
  return data.predictions as InferencePrediction[];
}

export interface CustomPrediction {
  home_team: string;
  away_team: string;
  predicted_outcome: string;
  confidence: number;
  probabilities: { home_win: number; draw: number; away_win: number };
  league: string;
  /** Set only when the two sides come from different leagues. */
  away_league?: string | null;
  /**
   * Which model produced the numbers. Absent (or null) is the domestic
   * ensemble; a cross-league tie is priced by ELO — or an ELO/Dixon-Coles
   * blend — because the ensemble has never seen a fixture spanning two
   * leagues. Worth showing: the two are not interchangeable.
   */
  model?: string | null;
}

/**
 * One hypothetical fixture.
 *
 * `awayLeagueCode` names the away side's league when it differs from the home
 * side's. Omitted means the same league, which is what a single code has
 * always meant here — the backend decides which model that implies.
 */
export async function predictCustom(
  homeTeam: string,
  awayTeam: string,
  leagueCode: string,
  awayLeagueCode?: string,
): Promise<CustomPrediction> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/predictions/custom`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      home_team: homeTeam,
      away_team: awayTeam,
      league_code: leagueCode,
      away_league_code: awayLeagueCode ?? null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    /*
     * A 422 here is usually a *stated refusal* — a club with no history to
     * predict from — and that sentence is the whole value of the response, so
     * it must survive to the screen. FastAPI's own validation errors share the
     * status but put a list of objects in `detail`, which would render as
     * "[object Object]"; only a string is worth showing.
     */
    const detail = typeof err.detail === 'string' ? err.detail : '';
    throw new Error(detail || 'Prediction failed');
  }
  return res.json() as Promise<CustomPrediction>;
}

export async function fetchTeams(): Promise<Record<string, string[]>> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/teams`, { headers });
  if (!res.ok) throw new Error('Failed to fetch teams');
  return res.json() as Promise<Record<string, string[]>>;
}

// ── Statistics ────────────────────────────────────────────────────────────────

/** Win/draw/loss counters plus the ratios derived from them. */
export interface TeamRecord {
  played: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
  points: number;
  points_per_game: number;
  win_pct: number;
  draw_pct: number;
  loss_pct: number;
  avg_goals_for: number;
  avg_goals_against: number;
  goal_difference: number;
}

/** One past match; `result`/`venue` are relative to the queried team. */
export interface MatchRecord {
  date: string;
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
  result: string | null;
  venue: string | null;
}

export interface TeamRates {
  clean_sheets: number;
  failed_to_score: number;
  btts: number;
  over_2_5: number;
}

export interface TeamAverages {
  shots: number;
  shots_on_target: number;
  corners: number;
  cards: number;
}

export interface TeamStats {
  team: string;
  league: string;
  league_code: string;
  overall: TeamRecord;
  home: TeamRecord;
  away: TeamRecord;
  recent: TeamRecord;
  form_sequence: string[];
  rates: TeamRates;
  averages: TeamAverages;
  recent_matches: MatchRecord[];
}

export interface HeadToHead {
  home_team: string;
  away_team: string;
  played: number;
  home_wins: number;
  draws: number;
  away_wins: number;
  home_goals: number;
  away_goals: number;
  avg_goals_home: number;
  avg_goals_away: number;
  avg_goals_total: number;
  btts_pct: number;
  over_2_5_pct: number;
  matches: MatchRecord[];
}

export interface GoalMarkets {
  expected_goals: { home: number; away: number; total: number };
  over_under: {
    over_1_5: number;
    over_2_5: number;
    over_3_5: number;
    under_2_5: number;
  };
  btts: { yes: number; no: number };
  top_scorelines: { score: string; prob: number }[];
  /** 'model' = calibrated Dixon-Coles; 'historical' = rates-based estimate. */
  source: string;
}

export interface MatchStats {
  home_team: string;
  away_team: string;
  league: string;
  league_code: string;
  head_to_head: HeadToHead;
  home: TeamStats;
  away: TeamStats;
  goal_markets: GoalMarkets | null;
}

export async function fetchMatchStats(
  homeTeam: string,
  awayTeam: string,
  leagueCode: string,
): Promise<MatchStats> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/stats/match`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, league_code: leagueCode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Estatísticas indisponíveis' }));
    throw new Error(err.detail || 'Estatísticas indisponíveis');
  }
  return res.json() as Promise<MatchStats>;
}

export async function fetchTeamStats(
  team: string,
  leagueCode: string,
): Promise<TeamStats> {
  const headers = await authHeaders();
  const params = new URLSearchParams({ team, league_code: leagueCode });
  const res = await fetch(`${BASE}/api/stats/team?${params}`, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Estatísticas indisponíveis' }));
    throw new Error(err.detail || 'Estatísticas indisponíveis');
  }
  return res.json() as Promise<TeamStats>;
}

// ── Retrain status ────────────────────────────────────────────────────────────

export async function fetchRetrainStatus(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/api/retrain-status`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.retraining as boolean;
  } catch {
    return false;
  }
}

// ── Auth / Registration ───────────────────────────────────────────────────────

export async function registerUser(email: string, password: string): Promise<void> {
  const { data, error } = await supabase.auth.signUp({ email, password });
  if (error) throw new Error(error.message);

  const token = data.session?.access_token;
  const userId = data.user?.id;

  // Build payload: use token when session is immediate, user_id when email confirmation is pending
  const payload = token ? { token } : userId ? { user_id: userId } : null;
  if (!payload) return;

  const res = await fetch(`${BASE}/api/users/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(err.detail || 'Registration failed');
  }

  // Sign out immediately — user must wait for admin approval
  await supabase.auth.signOut();
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export interface AdminUser {
  user_id: string;
  email: string;
  approved: boolean;
  is_admin: boolean;
}

export async function adminListUsers(): Promise<AdminUser[]> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/users`, { headers });
  if (!res.ok) throw new Error('Failed to fetch users');
  return res.json();
}

export async function adminCreateUser(email: string, password: string): Promise<AdminUser> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to create user' }));
    throw new Error(err.detail || 'Failed to create user');
  }
  return res.json();
}

// ── Admin: canonical team-name aliases ────────────────────────────────────────

export interface PendingTeamAlias {
  league_code: string;
  raw_name: string;
  /** Advisory candidates — an admin picks one, nothing is auto-applied. */
  suggestions: string[];
  /**
   * Country of a name from a UEFA competition, whose scope encodes it
   * ("EU-POR"). Null for domestic names.
   */
  country: string | null;
  /**
   * Every team this entry may be mapped to. Sent per entry because a UEFA
   * scope is not a league, so it cannot be looked up in `teams`.
   */
  options: string[];
}

export interface ApprovedTeamAlias {
  league_code: string;
  raw_name: string;
  canonical_name: string;
}

export interface TeamAliasList {
  approved: ApprovedTeamAlias[];
  pending: PendingTeamAlias[];
  /** Canonical names per league, so the picker only offers real teams. */
  teams: Record<string, string[]>;
}

export async function adminListTeamAliases(): Promise<TeamAliasList> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/team-aliases`, { headers });
  if (!res.ok) throw new Error('Falha ao carregar nomes de equipas');
  return res.json() as Promise<TeamAliasList>;
}

export async function adminApproveTeamAlias(
  leagueCode: string,
  rawName: string,
  canonicalName: string,
): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/team-aliases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({
      league_code: leagueCode,
      raw_name: rawName,
      canonical_name: canonicalName,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Falha ao aprovar' }));
    throw new Error(err.detail || 'Falha ao aprovar');
  }
}

export async function adminRevokeTeamAlias(
  leagueCode: string,
  rawName: string,
): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/team-aliases`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ league_code: leagueCode, raw_name: rawName }),
  });
  if (!res.ok) throw new Error('Falha ao remover');
}

export async function adminSetApproved(userId: string, approved: boolean): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/users/${userId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ approved }),
  });
  if (!res.ok) throw new Error('Failed to update user');
}
