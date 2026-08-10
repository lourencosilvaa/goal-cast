import clsx from 'clsx';
import { AlertTriangle, Loader2, Radio } from 'lucide-react';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { useLiveResults } from '@/hooks/useLiveResults';
import type { LiveMatch, LiveResults } from '@/lib/api';

const STATUS_LABEL: Record<string, string> = {
  scheduled: 'Por começar',
  live: 'Ao vivo',
  paused: 'Intervalo',
  finished: 'Terminado',
  postponed: 'Adiado',
  cancelled: 'Cancelado',
};

/** Statuses whose matches are worth putting at the top of the board. */
const IN_PROGRESS = new Set(['live', 'paused']);

/**
 * Ordering: what is happening now, then what is about to, then what is done.
 * Within a group, by kick-off — the natural reading order of a matchday.
 */
const STATUS_RANK: Record<string, number> = {
  live: 0,
  paused: 1,
  scheduled: 2,
  finished: 3,
  postponed: 4,
  cancelled: 5,
};

function ordered(matches: LiveMatch[]): LiveMatch[] {
  return [...matches].sort((a, b) => {
    const byStatus = (STATUS_RANK[a.status] ?? 9) - (STATUS_RANK[b.status] ?? 9);
    return byStatus !== 0 ? byStatus : a.kickoff.localeCompare(b.kickoff);
  });
}

function kickoffTime(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? ''
    : parsed.toLocaleTimeString('pt-PT', { hour: '2-digit', minute: '2-digit' });
}

/**
 * What to show in the clock column.
 *
 * A provider clock is printed as it came. A derived one is prefixed with "~"
 * so nobody reads arithmetic as the official minute — it cannot see stoppage
 * time or a delayed start. Before kick-off the column shows the kick-off time
 * instead, which is the only thing anyone wants there.
 */
function clockLabel(match: LiveMatch): string {
  if (match.status === 'scheduled') return kickoffTime(match.kickoff);
  if (match.status === 'paused') return 'Int.';
  if (match.status === 'finished') return 'Final';
  if (!IN_PROGRESS.has(match.status)) return STATUS_LABEL[match.status] ?? '';
  if (match.minute) return match.minute;
  return match.minute_estimated ? `~${match.elapsed_minutes}'` : `${match.elapsed_minutes}'`;
}

function score(match: LiveMatch): string {
  if (match.home_goals === null || match.away_goals === null) return '–';
  return `${match.home_goals}-${match.away_goals}`;
}

/** Matches whose score changed on the last refresh, for a brief highlight. */
function scoredKeys(results: LiveResults): Set<string> {
  return new Set(
    results.events
      .filter((event) => event.type === 'goal')
      .map((event) => `${event.match.home_team}|${event.match.away_team}`),
  );
}

function MatchLine({ match, justScored }: { match: LiveMatch; justScored: boolean }) {
  const inPlay = IN_PROGRESS.has(match.status);
  return (
    <li
      className={clsx(
        'grid grid-cols-[3rem_1fr_auto] items-center gap-3 py-2 text-sm',
        'border-b border-line-soft last:border-b-0',
        justScored && 'bg-accent-green/10',
      )}
    >
      <span
        className={clsx(
          'font-mono text-xs tabular-nums',
          inPlay ? 'text-accent-green' : 'text-fg-subtle',
        )}
        title={
          match.minute_estimated
            ? 'Minuto estimado a partir da hora de início — a fonte não o publica'
            : undefined
        }
      >
        {clockLabel(match)}
      </span>

      <span className="truncate">
        {match.home_team} <span className="text-fg-subtle">vs</span> {match.away_team}
      </span>

      <span
        className={clsx(
          'font-mono tabular-nums',
          inPlay ? 'text-fg font-semibold' : 'text-fg-muted',
        )}
      >
        {score(match)}
      </span>
    </li>
  );
}

export interface LiveScoreboardProps {
  /** League codes to show; omit for every league the backend offers. */
  leagues?: string[];
}

/**
 * Today's scores, from the dedicated results service.
 *
 * Three states are kept visibly distinct, because collapsing any two of them
 * tells the user something untrue: an empty board ("nothing today"), a stale
 * board ("these were the scores, we cannot currently refresh") and an outright
 * failure ("the service is not answering").
 */
export function LiveScoreboard({ leagues }: LiveScoreboardProps) {
  const { data, loading, error } = useLiveResults(leagues);

  if (loading && !data) {
    return (
      <Panel padding="md">
        <div className="flex items-center gap-2 text-sm text-fg-muted">
          <Loader2 className="w-4 h-4 animate-spin text-accent" />A carregar resultados…
        </div>
      </Panel>
    );
  }

  if (error && !data) {
    return (
      <Panel padding="md" accent="red">
        <div className="flex items-start gap-2 text-sm text-fg-muted">
          <AlertTriangle className="w-4 h-4 text-accent-red shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      </Panel>
    );
  }

  if (!data) return null;

  const matches = ordered(data.matches);
  const goals = scoredKeys(data);
  const playing = matches.filter((match) => IN_PROGRESS.has(match.status)).length;

  return (
    <Panel padding="md" accent={playing > 0 ? 'green' : 'neutral'}>
      <div className="flex items-center justify-between gap-3 mb-3">
        <SectionLabel>
          <span className="inline-flex items-center gap-1.5">
            <Radio
              className={clsx('w-3.5 h-3.5', playing > 0 && 'text-accent-green')}
            />
            Resultados de hoje
          </span>
        </SectionLabel>

        {data.stale && (
          <span
            className="text-[11px] font-mono text-accent-amber"
            title="A fonte não responde; estes são os últimos resultados conhecidos"
          >
            desatualizado
          </span>
        )}
        {error && !data.stale && (
          <span className="text-[11px] font-mono text-accent-amber" title={error}>
            sem atualização
          </span>
        )}
      </div>

      {matches.length === 0 ? (
        <p className="text-sm text-fg-subtle">Nenhum jogo hoje nas ligas seguidas.</p>
      ) : (
        <ul>
          {matches.map((match) => (
            <MatchLine
              key={`${match.league}|${match.home_team}|${match.away_team}`}
              match={match}
              justScored={goals.has(`${match.home_team}|${match.away_team}`)}
            />
          ))}
        </ul>
      )}
    </Panel>
  );
}
