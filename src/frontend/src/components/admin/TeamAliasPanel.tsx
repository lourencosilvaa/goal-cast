import { useCallback, useEffect, useState } from 'react';
import { Check, Loader2, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { SectionLabel } from '@/components/ui/SectionLabel';
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
      <div className="flex items-center justify-center py-10">
        <Loader2 className="w-5 h-5 animate-spin text-fg-subtle" />
      </div>
    );
  }

  const pending = data?.pending ?? [];
  const approved = data?.approved ?? [];

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[13px] text-fg-muted leading-relaxed max-w-[640px]">
        Nomes recolhidos do FlashScore e das competições europeias que não correspondem a nenhuma
        equipa conhecida. Enquanto não forem validados, os jogos correspondentes são ignorados —
        nunca previstos com médias da liga. A sugestão no topo vem pré-seleccionada, mas nada é
        aplicado sem confirmação.
      </p>

      {error && <p className="text-xs text-accent-red">✗ {error}</p>}

      <div className="flex items-center gap-2">
        <SectionLabel>Por rever</SectionLabel>
        <span className="font-mono text-[11px] text-fg-subtle">{pending.length}</span>
      </div>

      {pending.length === 0 ? (
        <p className="text-xs text-fg-subtle">Nada por rever.</p>
      ) : (
        <div className="flex flex-col gap-2">
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
                className="flex flex-col sm:flex-row sm:items-center gap-2.5 px-4 py-3 rounded-lg border border-line-soft bg-card"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className="text-sm font-semibold text-fg truncate">{entry.raw_name}</span>
                  <span className="font-mono text-[10px] text-fg-subtle shrink-0">
                    {entry.league_code}
                  </span>
                  {entry.country && (
                    <span className="px-1.5 py-0.5 rounded border border-line font-mono text-[9px] text-fg-subtle shrink-0">
                      {entry.country}
                    </span>
                  )}
                </div>

                <select
                  value={selected}
                  onChange={(e) => setChoice({ ...choice, [id]: e.target.value })}
                  className="px-3 py-1.5 rounded-md bg-card border border-line text-fg text-xs outline-none focus:border-accent/45 sm:w-56 cursor-pointer"
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

                <Button
                  size="sm"
                  loading={busy === id}
                  disabled={!selected}
                  onClick={() => handleApprove(entry.league_code, entry.raw_name, selected)}
                >
                  <Check className="w-3.5 h-3.5" />
                  Validar
                </Button>
              </div>
            );
          })}
        </div>
      )}

      {approved.length > 0 && (
        <>
          <div className="flex items-center gap-2 mt-2">
            <SectionLabel>Validados</SectionLabel>
            <span className="font-mono text-[11px] text-fg-subtle">{approved.length}</span>
          </div>
          <div className="flex flex-col gap-2">
            {approved.map((entry) => {
              const id = key(entry.league_code, entry.raw_name);
              return (
                <div
                  key={id}
                  className="flex items-center gap-3 px-4 py-2.5 rounded-lg border border-line-soft bg-card"
                >
                  <span className="text-[13px] text-fg-muted truncate flex-1">
                    {entry.raw_name}
                  </span>
                  <span className="text-fg-subtle text-xs shrink-0">→</span>
                  <span className="text-[13px] text-accent truncate flex-1">
                    {entry.canonical_name}
                  </span>
                  <span className="font-mono text-[10px] text-fg-subtle shrink-0">
                    {entry.league_code}
                  </span>
                  <button
                    onClick={() => handleRevoke(entry.league_code, entry.raw_name)}
                    disabled={busy === id}
                    className="p-1.5 rounded text-fg-subtle hover:text-accent-red transition-colors disabled:opacity-40 cursor-pointer shrink-0"
                    title="Remover"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
