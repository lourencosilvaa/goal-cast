import { config } from '@/config';
import { supabase } from '@/lib/supabase';
import type { League, LeaguePredictions } from '@/types';

const BASE = config.apiUrl;

async function authHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

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

export async function fetchAllPredictions(date?: string): Promise<LeaguePredictions[]> {
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
  const res = await fetch(`${BASE}/api/predictions/${leagueCode}${params}`);
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function refreshPredictions(leagueCode?: string): Promise<void> {
  const params = leagueCode ? `?league_code=${leagueCode}` : '';
  await fetch(`${BASE}/api/predictions/refresh${params}`, { method: 'POST' });
}

export async function downloadExport(format: 'csv' | 'excel', date?: string): Promise<void> {
  const params = new URLSearchParams({ format });
  if (date) params.set('report_date', date);
  const res = await fetch(`${BASE}/api/export?${params}`);
  if (!res.ok) throw new Error('Export failed');
  const blob = await res.blob();
  const ext = format === 'excel' ? 'xlsx' : 'csv';
  const filename = `predictions_${date ?? 'latest'}.${ext}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function analyzeMatch(
  matchData: Record<string, unknown>,
  model?: string,
): Promise<string> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/ai/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({
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
  if (!token) {
    // Email confirmation required — still inform backend if we have a user id
    // but we can't create profile without a token; the user must confirm first.
    return;
  }

  const res = await fetch(`${BASE}/api/users/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
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

export async function adminSetApproved(userId: string, approved: boolean): Promise<void> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}/api/admin/users/${userId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ approved }),
  });
  if (!res.ok) throw new Error('Failed to update user');
}
