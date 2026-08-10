import clsx from 'clsx';
import { ArrowRight } from 'lucide-react';
import { ConfidenceBadge } from '@/components/ui/ConfidenceBadge';
import { ProbabilityBar, ProbabilityLegend } from '@/components/ui/ProbabilityBar';
import { InPlayBlock, inPlayClock } from '@/components/live/InPlayBlock';
import type { InPlayMatch } from '@/lib/api';
import type { MatchPrediction } from '@/types';
import { outcomeShorthand } from './derive';

/**
 * The one-line summary shown under the probability bar when a row is open.
 *
 * Prefers the strongest value bet, because that is the only line that carries
 * an actionable claim; falls back to the goal model, then to recent form.
 * Returns null when the writer recorded none of the three.
 */
function signalLine(match: MatchPrediction): string | null {
  const best = [...match.value_bets].sort((a, b) => b.edge - a.edge)[0];
  if (best) {
    return `Value: ${best.outcome} · edge ${best.edge_pct} · odds ${best.best_odds.toFixed(2)} · Kelly ${(best.kelly_fraction * 100).toFixed(1)}%`;
  }

  const xg = match.expected_goals;
  if (xg) {
    const over = match.over_under
      ? ` · Over 2.5 ${(match.over_under.over_25 * 100).toFixed(0)}%`
      : '';
    const btts = match.btts ? ` · BTTS ${(match.btts.yes * 100).toFixed(0)}%` : '';
    return `xG ${xg.home.toFixed(1)}–${xg.away.toFixed(1)} (total ${xg.total.toFixed(1)})${over}${btts}`;
  }

  if (match.form) {
    return `Forma (ppj): ${match.home_team} ${match.form.home.toFixed(1)} · ${match.away_team} ${match.form.away.toFixed(1)}`;
  }

  return null;
}

/** Bookmaker prices, when the writer captured them. */
function oddsLine(match: MatchPrediction): string | null {
  if (!match.odds) return null;
  return `B365  1 ${match.odds.home.toFixed(2)}   X ${match.odds.draw.toFixed(2)}   2 ${match.odds.away.toFixed(2)}`;
}

interface MatchRowProps {
  match: MatchPrediction;
  expanded: boolean;
  onToggle: () => void;
  /** Opens the full detail — docked panel on wide screens, sub-page below. */
  onOpenDetail: () => void;
  /**
   * The live re-pricing for this fixture, when it is being played *and* the
   * board could identify it. Absent is the normal case: most rows are not
   * live, and one that is may still have a club spelling nobody has approved.
   */
  inPlay?: InPlayMatch;
}

export function MatchRow({
  match,
  expanded,
  onToggle,
  onOpenDetail,
  inPlay,
}: MatchRowProps) {
  const signal = signalLine(match);
  const odds = oddsLine(match);
  const hasValue = match.value_bets.length > 0;

  return (
    <div className="border-b border-line-soft">
      <div
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onToggle();
          }
        }}
        className={clsx(
          'flex items-center gap-2.5 px-5 md:px-7 py-3.5 cursor-pointer outline-none',
          'hover:bg-card-2/60 transition-colors',
          expanded && 'bg-card-2',
        )}
      >
        <span className="w-12 shrink-0 font-mono text-[11px] text-fg-subtle">
          {match.time || '—'}
        </span>

        <span className="flex-1 flex items-center gap-2 min-w-0 whitespace-nowrap">
          <span className="text-sm font-bold text-fg truncate min-w-0">{match.home_team}</span>
          <span className="text-[11px] text-fg-subtle shrink-0">vs</span>
          <span className="text-sm font-bold text-fg truncate min-w-0">{match.away_team}</span>
          {inPlay && (
            <span
              className="shrink-0 px-1.5 py-0.5 rounded bg-accent-green/15 border border-accent-green/40 text-accent-green font-mono text-[9px] font-bold tabular-nums"
              title="Jogo a decorrer — abra a linha para a previsão ao vivo"
            >
              {inPlay.home_goals}-{inPlay.away_goals} · {inPlayClock(inPlay)}
            </span>
          )}
          {hasValue && (
            <span className="shrink-0 px-1.5 py-0.5 rounded border border-accent-green/40 text-accent-green font-mono text-[9px] font-bold">
              VALUE
            </span>
          )}
          {match.model && (
            <span
              className="shrink-0 px-1.5 py-0.5 rounded border border-line text-fg-subtle font-mono text-[9px] font-bold uppercase"
              title={`Previsto por ${match.model} — não pelo ensemble doméstico`}
            >
              {match.model}
            </span>
          )}
        </span>

        <span className="w-8 shrink-0 text-center font-mono text-sm font-bold text-fg">
          {outcomeShorthand(match)}
        </span>

        <ConfidenceBadge confidence={match.confidence} />
      </div>

      {/*
        The detail is indented to where the team names start (row padding +
        time column + gap), so it hangs off the fixture rather than the page
        edge. Written as pl/pr rather than px + pl: both write padding-left,
        and their precedence would come from Tailwind's ordering rules, not
        from the order they appear in this string.
      */}
      {expanded && (
        <div className="flex flex-col gap-3 pl-5 md:pl-[86px] pr-5 md:pr-7 pb-5">
          <div className="max-w-[520px]">
            {/* Labelled only when a live bar sits below it — on its own there
                is nothing for "pré-jogo" to distinguish it from. */}
            {inPlay && <p className="label-mono text-fg-subtle mb-1">Pré-jogo</p>}
            <ProbabilityLegend
              probabilities={match.probabilities}
              homeLabel={match.home_team}
              awayLabel={match.away_team}
              className="mb-1.5"
            />
            <ProbabilityBar probabilities={match.probabilities} />
          </div>

          {inPlay && <InPlayBlock match={inPlay} />}

          {signal && <p className="font-mono text-xs text-accent">{signal}</p>}
          {odds && <p className="font-mono text-xs text-fg-subtle">{odds}</p>}

          {match.top_scorelines && match.top_scorelines.length > 0 && (
            <div className="flex gap-1.5 flex-wrap">
              {match.top_scorelines.slice(0, 4).map((line) => (
                <span
                  key={line.score}
                  className="px-2 py-0.5 rounded border border-line font-mono text-[11px] text-fg-muted"
                >
                  {line.score} · {(line.prob * 100).toFixed(0)}%
                </span>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpenDetail();
            }}
            className="flex items-center gap-1.5 w-fit text-xs font-semibold text-fg-muted hover:text-accent transition-colors cursor-pointer"
          >
            Detalhes completos
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
