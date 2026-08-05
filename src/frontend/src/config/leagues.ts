/**
 * Leagues offered by the on-demand prediction and statistics pages.
 *
 * Kept in one place so the "Prever Jogo" and "Equipas" pickers can never drift
 * apart. Codes must match `data.leagues` in `config/config.yaml`.
 */
export interface LeagueOption {
  code: string;
  name: string;
  flag: string;
}

export const LEAGUES: readonly LeagueOption[] = [
  { code: 'E0', name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
  { code: 'SP1', name: 'La Liga', flag: '🇪🇸' },
  { code: 'D1', name: 'Bundesliga', flag: '🇩🇪' },
  { code: 'I1', name: 'Serie A', flag: '🇮🇹' },
  { code: 'F1', name: 'Ligue 1', flag: '🇫🇷' },
  { code: 'P1', name: 'Liga Portugal', flag: '🇵🇹' },
] as const;

export const DEFAULT_LEAGUE_CODE = LEAGUES[0].code;
