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
  { code: 'E1', name: 'Championship', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
  { code: 'E2', name: 'League One', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
  { code: 'E3', name: 'League Two', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
  { code: 'SC0', name: 'Scottish Premiership', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿' },
  { code: 'SC1', name: 'Scottish Championship', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿' },
  { code: 'SC2', name: 'Scottish League One', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿' },
  { code: 'SC3', name: 'Scottish League Two', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿' },
  { code: 'SP1', name: 'La Liga', flag: '🇪🇸' },
  { code: 'SP2', name: 'La Liga 2', flag: '🇪🇸' },
  { code: 'D1', name: 'Bundesliga', flag: '🇩🇪' },
  { code: 'D2', name: '2. Bundesliga', flag: '🇩🇪' },
  { code: 'I1', name: 'Serie A', flag: '🇮🇹' },
  { code: 'I2', name: 'Serie B', flag: '🇮🇹' },
  { code: 'F1', name: 'Ligue 1', flag: '🇫🇷' },
  { code: 'F2', name: 'Ligue 2', flag: '🇫🇷' },
  { code: 'N1', name: 'Eredivisie', flag: '🇳🇱' },
  { code: 'B1', name: 'Jupiler Pro League', flag: '🇧🇪' },
  { code: 'P1', name: 'Liga Portugal', flag: '🇵🇹' },
  { code: 'T1', name: 'Super Lig', flag: '🇹🇷' },
  { code: 'G1', name: 'Super League Greece', flag: '🇬🇷' },
] as const;

export const DEFAULT_LEAGUE_CODE = LEAGUES[0].code;
