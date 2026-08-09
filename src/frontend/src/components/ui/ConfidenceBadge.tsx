import clsx from 'clsx';
import { tierFor, type ConfidenceTier } from '@/config/theme';

const tierClass: Record<ConfidenceTier, string> = {
  high: 'bg-accent text-bg border-accent',
  medium: 'bg-transparent text-accent border-accent/45',
  low: 'bg-transparent text-fg-subtle border-line',
};

/**
 * Model confidence as a three-step scale. The fill weight carries the reading,
 * so a column of these is scannable without parsing every number.
 */
export function ConfidenceBadge({
  confidence,
  className,
}: {
  /** 0..1, as returned by the API. */
  confidence: number;
  className?: string;
}) {
  return (
    <span
      className={clsx(
        'inline-block w-11 shrink-0 text-center py-1 rounded border',
        'font-mono text-[11px] font-semibold',
        tierClass[tierFor(confidence)],
        className,
      )}
      title={`Confiança do modelo: ${(confidence * 100).toFixed(1)}%`}
    >
      {(confidence * 100).toFixed(0)}%
    </span>
  );
}
