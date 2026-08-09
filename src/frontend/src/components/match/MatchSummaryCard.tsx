import { motion } from 'framer-motion';
import clsx from 'clsx';
import { ChevronRight, TrendingUp } from 'lucide-react';
import type { MatchPrediction } from '@/types';

function pct(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

function outcomeShort(match: MatchPrediction): { label: string; color: string } {
  if (match.predicted_outcome === 'Home Win')
    return { label: '1', color: 'text-accent-green' };
  if (match.predicted_outcome === 'Away Win')
    return { label: '2', color: 'text-accent-red' };
  return { label: 'X', color: 'text-accent-amber' };
}

interface MatchSummaryCardProps {
  match: MatchPrediction;
  index: number;
  selected?: boolean;
  onSelect: () => void;
}

/** Compact, clickable summary row used in the center master list. */
export function MatchSummaryCard({ match, index, selected, onSelect }: MatchSummaryCardProps) {
  const p = match.probabilities;
  const hasValueBets = match.value_bets.length > 0;
  const outcome = outcomeShort(match);
  const bestEdge = hasValueBets
    ? Math.max(...match.value_bets.map((v) => v.edge))
    : 0;

  return (
    <motion.button
      type="button"
      onClick={onSelect}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03, duration: 0.25 }}
      className={clsx(
        'w-full text-left glass rounded-2xl border p-4 transition-all cursor-pointer',
        'hover:border-accent-blue/40',
        selected ? 'border-accent-blue/60 ring-1 ring-accent-blue/30' : 'border-line',
      )}
    >
      {/* Top row */}
      <div className="flex items-center gap-2 mb-3">
        {match.time && <span className="text-[11px] text-fg-subtle font-mono">{match.time}</span>}
        {hasValueBets && (
          <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-accent-green/15 text-accent-green border border-accent-green/25 font-medium">
            <TrendingUp className="w-3 h-3" />
            {(bestEdge).toFixed(1)}%
          </span>
        )}
        {/*
          Shown only when the writer recorded a model, which today means a
          cross-league fixture predicted by ratings alone. A domestic
          prediction carries no label and gets no badge, so the absence is
          what marks the default rather than a name that has to be kept in
          sync here.
        */}
        {match.model && (
          <span
            className="text-[10px] px-2 py-0.5 rounded-full bg-fg-subtle/10 text-fg-subtle border border-fg-subtle/20 font-medium uppercase tracking-wide"
            title={`Predicted by ${match.model} — not the domestic ensemble`}
          >
            {match.model}
          </span>
        )}
        <span className="ml-auto text-[11px] text-fg-subtle">{match.league}</span>
        <ChevronRight className="w-4 h-4 text-fg-subtle" />
      </div>

      {/* Teams */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-fg font-medium flex-1 truncate">{match.home_team}</span>
        <span className={clsx('px-2 text-sm font-bold', outcome.color)}>{outcome.label}</span>
        <span className="text-fg font-medium flex-1 truncate text-right">{match.away_team}</span>
      </div>

      {/* Mini probability strip */}
      <div className="flex h-1.5 rounded-full overflow-hidden bg-card-2">
        <div className="bg-accent-green" style={{ width: `${p.home_win * 100}%` }} />
        <div className="bg-accent-amber" style={{ width: `${p.draw * 100}%` }} />
        <div className="bg-accent-red" style={{ width: `${p.away_win * 100}%` }} />
      </div>
      <div className="flex justify-between mt-1.5 text-[10px] text-fg-subtle font-mono">
        <span>{pct(p.home_win)}</span>
        <span>{pct(p.draw)}</span>
        <span>{pct(p.away_win)}</span>
      </div>
    </motion.button>
  );
}
