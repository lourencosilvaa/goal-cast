import { useCallback, useEffect, useState } from 'react';
import { Check, Loader2, Tags, Trash2, TriangleAlert } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import {
  adminApproveTeamAlias,
  adminListTeamAliases,
  adminRevokeTeamAlias,
  type TeamAliasList,
} from '@/lib/api';

/**
 * Review queue for team names scraped from FlashScore.
 *
 * The pipeline queues any name it could not match to the canonical
 * football-data spelling and skips that fixture. Nothing is resolved until an
 * admin confirms it here — the suggestions are proposals, never decisions.
 */
export function TeamAliasPanel() {
  const [data, setData] = useState<TeamAliasList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [choice, setChoice] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);

  // Awaits before its first setState, so mounting this component does not
  // update state synchronously inside the effect (react-hooks/set-state-in-effect).
  const load = useCallback(async () => {
    try {
      const next = await adminListTeamAliases();
      setData(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro ao carregar nomes');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Fetch-on-mount. The project has no data-fetching library, so this is the
    // same effect-driven load every other page here uses; `load` awaits before
    // its first setState, so nothing is set during the synchronous effect body.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  const key = (leagueCode: string, rawName: string) => `${leagueCode}::${rawName}`;

  async function handleApprove(leagueCode: string, rawName: string) {
    const canonical = choice[key(leagueCode, rawName)];
    if (!canonical) return;
    setBusy(key(leagueCode, rawName));
    setError(null);
    try {
      await adminApproveTeamAlias(leagueCode, rawName, canonical);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro ao aprovar');
    } finally {
      setBusy(null);
    }
  }

  async function handleRevoke(leagueCode: string, rawName: string) {
    setBusy(key(leagueCode, rawName));
    setError(null);
    try {
      await adminRevokeTeamAlias(leagueCode, rawName);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro ao remover');
    } finally {
      setBusy(null);
    }
  }

  if (loading) {
    return (
      <GlassCard className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 animate-spin text-fg-subtle" />
      </GlassCard>
    );
  }

  const pending = data?.pending ?? [];
  const approved = data?.approved ?? [];

  return (
    <GlassCard className="space-y-5" overflow="visible">
      <div className="flex items-center gap-2">
        <Tags className="w-4 h-4 text-accent-blue" />
        <h2 className="text-sm font-semibold text-fg">Nomes de Equipas</h2>
        {pending.length > 0 && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-amber/15 text-accent-amber border border-accent-amber/25 font-medium">
            {pending.length} por rever
          </span>
        )}
      </div>

      <p className="text-xs text-fg-muted leading-relaxed">
        Nomes recolhidos do FlashScore que não correspondem a nenhuma equipa conhecida.
        Enquanto não forem validados, os jogos correspondentes são ignorados — nunca
        previstos com médias da liga.
      </p>

      {error && <p className="text-xs text-accent-red">✗ {error}</p>}

      {pending.length === 0 ? (
        <p className="text-xs text-fg-subtle">Nada por rever.</p>
      ) : (
        <div className="space-y-2">
          {pending.map((entry) => {
            const id = key(entry.league_code, entry.raw_name);
            const options = data?.teams[entry.league_code] ?? [];
            return (
              <div
                key={id}
                className="flex flex-col sm:flex-row sm:items-center gap-2 px-3 py-2.5 rounded-xl bg-card-2 border border-line"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <TriangleAlert className="w-3.5 h-3.5 text-accent-amber shrink-0" />
                  <span className="text-sm text-fg truncate">{entry.raw_name}</span>
                  <span className="text-[10px] text-fg-subtle shrink-0">
                    {entry.league_code}
                  </span>
                </div>
                <select
                  value={choice[id] ?? ''}
                  onChange={(e) => setChoice({ ...choice, [id]: e.target.value })}
                  className="px-3 py-1.5 rounded-lg bg-card border border-line text-fg text-xs focus:outline-none focus:border-accent-blue/40 sm:w-56"
                >
                  <option value="">Escolher equipa…</option>
                  {entry.suggestions.length > 0 && (
                    <optgroup label="Sugestões">
                      {entry.suggestions.map((name) => (
                        <option key={`s-${name}`} value={name}>
                          {name}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  <optgroup label="Todas as equipas">
                    {options.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </optgroup>
                </select>
                <NeonButton
                  variant="primary"
                  size="sm"
                  loading={busy === id}
                  disabled={!choice[id]}
                  onClick={() => handleApprove(entry.league_code, entry.raw_name)}
                >
                  <Check className="w-3.5 h-3.5 mr-1 inline" />
                  Validar
                </NeonButton>
              </div>
            );
          })}
        </div>
      )}

      {approved.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-line">
          <p className="text-[10px] text-fg-subtle uppercase tracking-wider">
            Validados ({approved.length})
          </p>
          {approved.map((entry) => {
            const id = key(entry.league_code, entry.raw_name);
            return (
              <div
                key={id}
                className="flex items-center gap-2 px-3 py-2 rounded-xl bg-card-2 border border-line"
              >
                <span className="text-sm text-fg-muted truncate flex-1">
                  {entry.raw_name}
                </span>
                <span className="text-fg-subtle text-xs">→</span>
                <span className="text-sm text-accent-green truncate flex-1">
                  {entry.canonical_name}
                </span>
                <span className="text-[10px] text-fg-subtle">{entry.league_code}</span>
                <button
                  onClick={() => handleRevoke(entry.league_code, entry.raw_name)}
                  disabled={busy === id}
                  className="p-1.5 rounded-lg text-fg-subtle hover:text-accent-red hover:bg-accent-red/10 transition-colors disabled:opacity-40"
                  title="Remover"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </GlassCard>
  );
}
