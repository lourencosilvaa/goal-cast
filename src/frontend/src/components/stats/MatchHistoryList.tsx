import type { MatchRecord } from '@/lib/api';
import { shortDate } from './format';

const RESULT_STYLE: Record<string, string> = {
  W: 'bg-accent-green/15 text-accent-green border-accent-green/25',
  D: 'bg-accent-amber/15 text-accent-amber border-accent-amber/25',
  L: 'bg-accent-red/15 text-accent-red border-accent-red/25',
};

const RESULT_LABEL: Record<string, string> = { W: 'V', D: 'E', L: 'D' };

/**
 * Chronological list of past matches, most recent first.
 *
 * `subject` is the team the `result`/`venue` flags belong to — its name is
 * emphasised so a row reads correctly whichever side it played on.
 */
export function MatchHistoryList({
  matches,
  subject,
  emptyLabel = 'Sem jogos registados',
}: {
  matches: MatchRecord[];
  subject?: string;
  emptyLabel?: string;
}) {
  if (matches.length === 0) {
    return <p className="text-xs text-fg-subtle">{emptyLabel}</p>;
  }

  return (
    <div className="space-y-1.5">
      {matches.map((match, i) => (
        <div
          key={`${match.date}-${i}`}
          className="flex items-center gap-3 px-3 py-2 rounded-xl bg-card-2 border border-line"
        >
          <span className="text-[10px] text-fg-subtle font-mono w-[68px] shrink-0">
            {shortDate(match.date)}
          </span>
          <span
            className={`text-sm flex-1 truncate text-right ${
              match.home_team === subject ? 'text-fg font-medium' : 'text-fg-muted'
            }`}
          >
            {match.home_team}
          </span>
          <span className="text-sm font-semibold text-fg font-mono shrink-0">
            {match.home_goals}-{match.away_goals}
          </span>
          <span
            className={`text-sm flex-1 truncate ${
              match.away_team === subject ? 'text-fg font-medium' : 'text-fg-muted'
            }`}
          >
            {match.away_team}
          </span>
          {match.result && (
            <span
              className={`w-5 h-5 rounded-md border text-[10px] font-semibold flex items-center justify-center shrink-0 ${
                RESULT_STYLE[match.result] ?? 'bg-card border-line text-fg-muted'
              }`}
            >
              {RESULT_LABEL[match.result] ?? match.result}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
