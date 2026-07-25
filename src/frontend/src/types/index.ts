export interface Probabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface Odds {
  home: number;
  draw: number;
  away: number;
}

export interface ExpectedGoals {
  home: number;
  away: number;
  total: number;
}

export interface OverUnder {
  over_15: number;
  over_25: number;
  over_35: number;
  under_25: number;
}

export interface Btts {
  yes: number;
  no: number;
}

export interface Scoreline {
  score: string;
  prob: number;
}

export interface Form {
  home: number;
  away: number;
}

export interface ValueBet {
  outcome: string;
  ml_probability: number;
  bookmaker_implied: number;
  edge: number;
  edge_pct: string;
  best_odds: number;
  kelly_fraction: number;
  confidence: string;
}

export interface MatchPrediction {
  home_team: string;
  away_team: string;
  league: string;
  time: string;
  probabilities: Probabilities;
  predicted_outcome: string;
  confidence: number;
  odds: Odds | null;
  implied_probabilities: Odds | null;
  expected_goals: ExpectedGoals | null;
  over_under: OverUnder | null;
  btts: Btts | null;
  top_scorelines: Scoreline[] | null;
  form: Form | null;
  value_bets: ValueBet[];
}

export interface LeaguePredictions {
  league_code: string;
  league_name: string;
  matches: MatchPrediction[];
}

export interface League {
  code: string;
  name: string;
}
