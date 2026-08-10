import clsx from 'clsx';
import { Radio } from 'lucide-react';
import { ProbabilityBar, ProbabilityLegend } from '@/components/ui/ProbabilityBar';
import type { InPlayMatch } from '@/lib/api';

/**
 * The live re-pricing, shown inside an expanded fixture row.
 *
 * It sits *below* the pre-match bar rather than replacing it, and that is the
 * whole design: a live 91% says little alone, but under a pre-match 72% it
 * says the match has swung since kick-off. Replacing the number would hide the
 * only interesting part.
 *
 * Everything the model conditioned on is printed alongside — score, minute,
 * minutes left, remaining expected goals — because a probability that moved
 * without a visible reason reads as a glitch.
 */

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

/** The clock, marked when it was derived rather than published. */
export function inPlayClock(match: InPlayMatch): string {
  if (match.status === 'paused') return 'Int.';
  return match.minute_estimated ? `~${match.elapsed_minutes}'` : `${match.elapsed_minutes}'`;
}

/**
 * How far the live number has moved from the pre-match one, for the outcome
 * the pre-match model favoured. Reported in percentage points: it is a
 * difference between two probabilities, and a percentage of a percentage is
 * the classic way to make that unreadable.
 */
function swing(match: InPlayMatch): { label: string; points: number; before: number } {
  const outcomes = [
    { label: match.home_team, before: match.pre_match.home_win, now: match.live.home_win },
    { label: 'Empate', before: match.pre_match.draw, now: match.live.draw },
    { label: match.away_team, before: match.pre_match.away_win, now: match.live.away_win },
  ];
  const favourite = outcomes.reduce((best, o) => (o.before > best.before ? o : best));
  return {
    label: favourite.label,
    points: Math.round((favourite.now - favourite.before) * 100),
    before: favourite.before,
  };
}

export function InPlayBlock({ match }: { match: InPlayMatch }) {
  const moved = swing(match);
  const sign = moved.points > 0 ? '+' : '';

  return (
    <div className="max-w-[520px] rounded border border-accent-green/30 bg-accent-green/[0.04] p-3">
      <div className="flex items-center gap-2 mb-2">
        <Radio className="w-3.5 h-3.5 text-accent-green" />
        <span className="label-mono text-accent-green">Ao vivo</span>
        <span className="font-mono text-xs tabular-nums text-fg font-semibold">
          {match.home_goals}-{match.away_goals}
        </span>
        <span
          className="font-mono text-[11px] tabular-nums text-fg-subtle"
          title={
            match.minute_estimated
              ? 'Minuto estimado a partir da hora de início — a fonte não o publica'
              : undefined
          }
        >
          {inPlayClock(match)}
        </span>
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          restam {match.remaining_minutes}&apos;
        </span>
      </div>

      <ProbabilityLegend
        probabilities={match.live}
        homeLabel={match.home_team}
        awayLabel={match.away_team}
        className="mb-1.5"
      />
      <ProbabilityBar probabilities={match.live} />

      <p className="mt-2 font-mono text-[11px] text-fg-subtle">
        {moved.points !== 0 && (
          <>
            <span
              className={clsx(moved.points > 0 ? 'text-accent-green' : 'text-accent-red')}
              title={`${moved.label} estava a ${pct(moved.before)} antes do apito inicial`}
            >
              {moved.label} {sign}
              {moved.points} pp
            </span>
            {' · '}
          </>
        )}
        xG final {match.expected_home_goals.toFixed(1)}–{match.expected_away_goals.toFixed(1)}
      </p>
    </div>
  );
}
