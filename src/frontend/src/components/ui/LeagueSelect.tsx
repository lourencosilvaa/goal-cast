import { flagFor } from '@/config/leagues';
import { useLeagues } from '@/hooks/useLeagues';
import { FIELD_CLASS } from './TeamCombobox';

/**
 * League picker shared by the prediction and statistics pages.
 *
 * Offers what the backend serves, never a compile-time list — that is what
 * kept withdrawn divisions on screen after they were removed from
 * `data.served_leagues`. There is deliberately no local fallback: if the list
 * cannot be loaded, an empty picker is correct and a stale one is not.
 */
export function LeagueSelect({
  value,
  onChange,
  label = 'Liga',
}: {
  value: string;
  onChange: (code: string) => void;
  label?: string;
}) {
  const { domestic, loading, error } = useLeagues();

  return (
    <div className="space-y-1.5">
      <label className="label-mono text-fg-subtle block">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading || !!error || domestic.length === 0}
        className={`${FIELD_CLASS} w-full cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {loading && <option value="">A carregar ligas…</option>}
        {!loading && !error && domestic.length === 0 && (
          <option value="">Nenhuma liga disponível</option>
        )}
        {error && <option value="">Ligas indisponíveis</option>}
        {domestic.map((league) => (
          <option key={league.code} value={league.code} className="bg-card text-fg">
            {flagFor(league.code)} {league.name}
          </option>
        ))}
      </select>
      {error && <p className="text-xs text-accent-red">{error}</p>}
    </div>
  );
}
