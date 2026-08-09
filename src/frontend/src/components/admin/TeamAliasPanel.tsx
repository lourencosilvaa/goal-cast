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
 * Review queue for team names the pipeline could not match.
 *
 * Two sources feed it: FlashScore fixtures, and the openfootball corpus behind
 * the European competitions. Either way the fixture is skipped until a human
 * confirms the mapping — the suggestions are proposals, never decisions.
 *
 * European entries carry a country ("EU-POR"), which narrows both the
 * suggestions and the picker to that country's teams. Without it a full name
 * like "AC Sparta Praha" draws confident nonsense from 21 leagues.
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

  async function handleApprove(leagueCode: string, rawName: string, canonical: string) {
    // Taken from the caller, not from `choice`: a pre-selected suggestion the
    // admin accepted without touching the dropdown has no entry there yet.
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
        Nomes recolhidos do FlashScore e das competições europeias que não
        correspondem a nenhuma equipa conhecida. Enquanto não forem validados, os
        jogos correspondentes são ignorados — nunca previstos com médias da liga.
        A sugestão no topo vem pré-seleccionada, mas nada é aplicado sem confirmação.
      </p>

      {error && <p className="text-xs text-accent-red">✗ {error}</p>}

      {pending.length === 0 ? (
        <p className="text-xs text-fg-subtle">Nada por rever.</p>
      ) : (
        <div className="space-y-2">
          {pending.map((entry) => {
            const id = key(entry.league_code, entry.raw_name);
            // Sent per entry: a UEFA scope ("EU-POR") is a country, not a
            // league, so it has no entry in `teams`.
            const options = entry.options.length
              ? entry.options
              : (data?.teams[entry.league_code] ?? []);
            // The top suggestion is pre-selected to save a click, never
            // applied — approval still needs the button. Suggestions are
            // routinely wrong in ways only a human catches ("Sport Lisboa e
            // Benfica" ranks "Sp Lisbon", which is Sporting, above "Benfica").
            const selected = choice[id] ?? entry.suggestions[0] ?? '';
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
                  {entry.country && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-blue/15 text-accent-blue border border-accent-blue/25 shrink-0">
                      {entry.country}
                    </span>
                  )}
                </div>
                <select
                  value={selected}
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
                  disabled={!selected}
                  onClick={() =>
                    handleApprove(entry.league_code, entry.raw_name, selected)
                  }
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
