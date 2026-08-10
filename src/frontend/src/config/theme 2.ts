/**
 * Presentation constants for the Matchline layout.
 *
 * Anything a component would otherwise hardcode as a magic number lives here,
 * so the shell can be re-proportioned or the confidence banding re-tuned
 * without hunting through JSX.
 */

/** Fixed widths of the two rails, in pixels. */
export const rail = {
  /** Left navigation rail (desktop only). */
  navWidthPx: 210,
  /** Dashboard statistics rail, right of the nav and left of the match list. */
  dashboardWidthPx: 190,
} as const;

/**
 * Confidence banding for the badge on a match row.
 *
 * `high` renders as a filled accent chip, `medium` as an outlined one and
 * anything below as a muted outline — so scanning the column reads as a
 * three-step scale rather than sixteen distinct numbers.
 */
export const confidenceTier = {
  highPct: 75,
  mediumPct: 60,
} as const;

export type ConfidenceTier = 'high' | 'medium' | 'low';

/** Bands a 0..1 confidence into one of the three display tiers. */
export function tierFor(confidence: number): ConfidenceTier {
  const asPct = confidence * 100;
  if (asPct >= confidenceTier.highPct) return 'high';
  if (asPct >= confidenceTier.mediumPct) return 'medium';
  return 'low';
}

/** Signals ticker. */
export const ticker = {
  /** Seconds for one full pass of the doubled list. */
  durationSeconds: 34,
  /** A prediction only becomes a signal above this confidence (0..1). */
  minConfidence: 0.7,
  /** Cap, so a busy fixture list does not produce a minutes-long loop. */
  maxItems: 12,
} as const;

/** Date pills on the dashboard filter bar. */
export const dateFilter = {
  /** Most-recent dates offered as pills; the rest stay in the overflow select. */
  maxPills: 6,
} as const;
