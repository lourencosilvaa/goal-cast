import clsx from 'clsx';
import type { Probabilities } from '@/types';

/**
 * The 1X2 distribution as one continuous bar.
 *
 * Home and away take the two accents; the draw takes the inert middle colour,
 * because a draw is not a better or worse outcome than either win — it is the
 * absence of one.
 */
export function ProbabilityBar({
  probabilities,
  className,
  height = 'md',
}: {
  probabilities: Probabilities;
  className?: string;
  height?: 'sm' | 'md';
}) {
  const p = probabilities;
  return (
    <div
      className={clsx(
        'flex rounded-sm overflow-hidden bg-card-2',
        height === 'sm' ? 'h-1.5' : 'h-2.5',
        className,
      )}
    >
      <div className="bg-accent" style={{ width: `${p.home_win * 100}%` }} />
      <div className="bg-neutral-bar" style={{ width: `${p.draw * 100}%` }} />
      <div className="bg-accent-2" style={{ width: `${p.away_win * 100}%` }} />
    </div>
  );
}

/** The three figures underneath a `ProbabilityBar`, aligned to its segments. */
export function ProbabilityLegend({
  probabilities,
  homeLabel,
  awayLabel,
  className,
}: {
  probabilities: Probabilities;
  homeLabel: string;
  awayLabel: string;
  className?: string;
}) {
  const p = probabilities;
  const asPct = (value: number) => `${(value * 100).toFixed(0)}%`;
  return (
    <div
      className={clsx(
        'flex justify-between gap-3 font-mono text-[11px] text-fg-subtle',
        className,
      )}
    >
      <span className="truncate">
        {homeLabel} <span className="text-fg-muted">{asPct(p.home_win)}</span>
      </span>
      <span className="shrink-0">
        EMPATE <span className="text-fg-muted">{asPct(p.draw)}</span>
      </span>
      <span className="truncate text-right">
        {awayLabel} <span className="text-fg-muted">{asPct(p.away_win)}</span>
      </span>
    </div>
  );
}
