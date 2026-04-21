import { config } from '@/config';
import type { League, LeaguePredictions } from '@/types';

const BASE = config.apiUrl;

export async function fetchAvailableDates(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/dates`);
  if (!res.ok) throw new Error('Failed to fetch dates');
  return res.json();
}

export async function fetchLeagues(): Promise<League[]> {
  const res = await fetch(`${BASE}/api/leagues`);
  if (!res.ok) throw new Error('Failed to fetch leagues');
  return res.json();
}

export async function fetchAllPredictions(
  date?: string,
): Promise<LeaguePredictions[]> {
  const params = date ? `?date=${encodeURIComponent(date)}` : '';
  const res = await fetch(`${BASE}/api/predictions${params}`);
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function fetchLeaguePredictions(
  leagueCode: string,
  date?: string,
): Promise<LeaguePredictions> {
  const params = date ? `?date=${encodeURIComponent(date)}` : '';
  const res = await fetch(
    `${BASE}/api/predictions/${leagueCode}${params}`,
  );
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function refreshPredictions(
  leagueCode?: string,
): Promise<void> {
  const params = leagueCode ? `?league_code=${leagueCode}` : '';
  await fetch(`${BASE}/api/predictions/refresh${params}`, {
    method: 'POST',
  });
}

export async function analyzeMatch(
  apiKey: string,
  matchData: Record<string, unknown>,
  model?: string,
): Promise<string> {
  const res = await fetch(`${BASE}/api/ai/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gemini_api_key: apiKey,
      match_data: matchData,
      language: 'pt-PT',
      ...(model && { model }),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'AI analysis failed' }));
    throw new Error(err.detail || 'AI analysis failed');
  }
  const data = await res.json();
  return data.analysis;
}
