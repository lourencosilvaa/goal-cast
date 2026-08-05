/**
 * Formatting helpers shared by the statistics components.
 *
 * Kept apart from `primitives.tsx` so that file exports components only —
 * mixing the two breaks React Fast Refresh (and the lint rule guarding it).
 */

export type Accent = 'green' | 'blue' | 'amber' | 'red' | 'purple';

/** Formats a 0..1 ratio as a percentage string. */
export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Formats a number with a fixed number of decimals. */
export function num(value: number, digits = 2): string {
  return value.toFixed(digits);
}

/** Renders an ISO date as dd/mm/yyyy. */
export function shortDate(iso: string): string {
  const [year, month, day] = iso.split('-');
  return year && month && day ? `${day}/${month}/${year}` : iso;
}

/** Accent colour for a points-per-game figure. */
export function formTone(pointsPerGame: number): Accent {
  if (pointsPerGame >= 2) return 'green';
  if (pointsPerGame >= 1.5) return 'amber';
  return 'red';
}
