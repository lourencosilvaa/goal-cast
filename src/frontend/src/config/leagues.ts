/**
 * Presentation for leagues. **Not** the list of them.
 *
 * The set of selectable leagues comes from `GET /api/leagues`, which reads
 * `data.served_leagues` in `config/config.yaml`. Keeping a second copy here
 * is what previously let the frontend go on offering divisions the backend
 * had already withdrawn.
 *
 * Flags stay client-side on purpose: an emoji is presentation, not
 * environment configuration, and a league missing from this map still renders
 * — it just gets the neutral glyph. So adding a league to the backend config
 * needs no frontend change at all.
 */

/** Neutral stand-in for a league with no flag mapped yet. */
export const FALLBACK_FLAG = '🏟️';

const LEAGUE_FLAGS: Record<string, string> = {
  E0: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  E1: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  E2: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  E3: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  SC0: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  SC1: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  SC2: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  SC3: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  SP1: '🇪🇸',
  SP2: '🇪🇸',
  D1: '🇩🇪',
  D2: '🇩🇪',
  I1: '🇮🇹',
  I2: '🇮🇹',
  F1: '🇫🇷',
  F2: '🇫🇷',
  N1: '🇳🇱',
  B1: '🇧🇪',
  P1: '🇵🇹',
  T1: '🇹🇷',
  G1: '🇬🇷',
  CL: '⭐',
  EL: '🟠',
  UECL: '🔵',
  ECL: '🔵',
};

/**
 * Flag for a league code.
 *
 * Codes not in the map are expected, not exceptional — the backend may serve
 * a league this file has never heard of.
 */
export function flagFor(code: string): string {
  return LEAGUE_FLAGS[code] ?? FALLBACK_FLAG;
}
