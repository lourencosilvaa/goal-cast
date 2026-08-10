import { useState } from 'react';
import { Loader2, Wand2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { LeagueSelect } from '@/components/ui/LeagueSelect';
import { TeamCombobox } from '@/components/ui/TeamCombobox';
import { ProbabilityBar, ProbabilityLegend } from '@/components/ui/ProbabilityBar';
import { PageBody } from '@/components/layout/PageBody';
import { HeadToHeadPanel } from '@/components/stats/HeadToHeadPanel';
import { GoalMarketsPanel } from '@/components/stats/GoalMarketsPanel';
import { TeamComparison } from '@/components/stats/TeamComparison';
import { useLeagues } from '@/hooks/useLeagues';
import { useTeams } from '@/hooks/useTeams';
import {
  fetchMatchStats,
  predictCustom,
  type CustomPrediction,
  type MatchStats,
} from '@/lib/api';

const OUTCOME_LABELS: Record<string, string> = {
  'Home Win': 'Vitória Casa',
  Draw: 'Empate',
  'Away Win': 'Vitória Fora',
};

const OUTCOME_SHORTHAND: Record<string, string> = {
  'Home Win': '1',
  Draw: 'X',
  'Away Win': '2',
};

export function CustomPredictPage() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [leagueCode, setLeagueCode] = useState('');
  const [awayLeagueCode, setAwayLeagueCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CustomPrediction | null>(null);
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { domestic } = useLeagues();
  const { teamsFor, error: teamsError } = useTeams();

  /*
   * Derived rather than stored: the default is the first league the backend
   * offers, which is not known at mount. Deriving avoids an effect that would
   * write state as soon as the list arrives.
   *
   * The away league defaults to the home one, so the page opens on the
   * ordinary same-league case and the second picker only matters when moved.
   * Both read `domestic`, which excludes the UEFA competitions — those have no
   * football-data division behind them, so they are not a side of a fixture.
   */
  const activeLeague = leagueCode || domestic[0]?.code || '';
  const activeAwayLeague = awayLeagueCode || activeLeague;
  const homeTeams = teamsFor(activeLeague);
  const awayTeams = teamsFor(activeAwayLeague);
  const isCrossLeague = activeAwayLeague !== activeLeague;

  function clearResults() {
    setResult(null);
    setStats(null);
    setStatsError(null);
    setError(null);
  }

  function handleLeagueChange(code: string) {
    setLeagueCode(code);
    setHomeTeam('');
    // The away side is cleared too when it shares the league: its list is
    // about to change under it. When the leagues differ it is untouched, since
    // nothing about that side moved.
    if (!isCrossLeague) setAwayTeam('');
    clearResults();
  }

  function handleAwayLeagueChange(code: string) {
    setAwayLeagueCode(code);
    setAwayTeam('');
    clearResults();
  }

  async function handlePredict() {
    const home = homeTeam.trim();
    const away = awayTeam.trim();
    if (!home || !away || !activeLeague) return;
    setLoading(true);
    clearResults();

    /*
     * The statistics are skipped entirely for a cross-league pairing rather
     * than requested and allowed to fail. `/match-insights` is scoped to one
     * league and answers 404 when either club is unknown there, so the call is
     * guaranteed to fail — firing it anyway would spend a round trip to
     * produce an error message that says nothing true about the fixture.
     *
     * The two reads stay independent otherwise: a statistics failure must
     * never hide a successful prediction.
     */
    const [prediction, matchStats] = await Promise.allSettled([
      predictCustom(home, away, activeLeague, isCrossLeague ? activeAwayLeague : undefined),
      isCrossLeague ? Promise.resolve(null) : fetchMatchStats(home, away, activeLeague),
    ]);

    if (prediction.status === 'fulfilled') {
      setResult(prediction.value);
    } else {
      setError(
        prediction.reason instanceof Error ? prediction.reason.message : 'Erro ao obter previsão',
      );
    }

    if (matchStats.status === 'fulfilled') {
      setStats(matchStats.value);
    } else {
      setStatsError(
        matchStats.reason instanceof Error
          ? matchStats.reason.message
          : 'Estatísticas indisponíveis',
      );
    }

    setLoading(false);
  }

  return (
    <PageBody
      maxWidth="md"
      intro="Escolhe duas equipas e gera a previsão do modelo para um confronto hipotético."
    >
      <Panel overflow="visible" className="flex flex-col gap-4">
        <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-3 items-end">
          <LeagueSelect
            label="Liga (casa)"
            value={activeLeague}
            onChange={handleLeagueChange}
          />
          <span className="hidden sm:block pb-3 text-xs text-fg-subtle">vs</span>
          <LeagueSelect
            label="Liga (fora)"
            value={activeAwayLeague}
            onChange={handleAwayLeagueChange}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-3 items-end">
          <TeamCombobox
            label="Casa"
            value={homeTeam}
            onChange={setHomeTeam}
            teams={homeTeams.filter((team) => isCrossLeague || team !== awayTeam)}
            placeholder="Selecionar…"
          />
          <span className="hidden sm:block pb-3 text-xs text-fg-subtle">vs</span>
          <TeamCombobox
            label="Fora"
            value={awayTeam}
            onChange={setAwayTeam}
            teams={awayTeams.filter((team) => isCrossLeague || team !== homeTeam)}
            placeholder="Selecionar…"
          />
        </div>

        {isCrossLeague && (
          <p className="text-[11px] text-fg-subtle">
            Confronto entre ligas diferentes: previsto por ELO calibrado entre ligas, não
            pelo ensemble doméstico — que nunca viu um jogo destes. Sem estatísticas nem
            histórico direto.
          </p>
        )}

        {teamsError && <p className="text-xs text-accent-red">{teamsError}</p>}

        <Button
          onClick={handlePredict}
          loading={loading}
          disabled={!homeTeam.trim() || !awayTeam.trim()}
          className="w-full"
        >
          <Wand2 className="w-4 h-4" />
          Gerar previsão
        </Button>
      </Panel>

      {error && (
        <Panel accent="red" padding="sm" className="mt-4">
          <p className="text-sm text-accent-red">{error}</p>
        </Panel>
      )}

      {result && (
        <Panel className="mt-4 flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <span className="text-xl font-extrabold text-fg truncate">{result.home_team}</span>
            <span className="font-mono text-3xl font-extrabold text-accent shrink-0">
              {OUTCOME_SHORTHAND[result.predicted_outcome] ?? '·'}
            </span>
            <span className="text-xl font-extrabold text-fg truncate">{result.away_team}</span>
            <span className="ml-auto text-right shrink-0">
              <span className="block font-mono text-xl font-bold text-fg">
                {(result.confidence * 100).toFixed(0)}%
              </span>
              <SectionLabel>Confiança</SectionLabel>
            </span>
          </div>

          <div>
            <ProbabilityLegend
              probabilities={result.probabilities}
              homeLabel={result.home_team}
              awayLabel={result.away_team}
              className="mb-1.5"
            />
            <ProbabilityBar probabilities={result.probabilities} />
          </div>

          <p className="font-mono text-xs text-accent">
            {result.away_league ? `${result.league} × ${result.away_league}` : result.league} ·{' '}
            {OUTCOME_LABELS[result.predicted_outcome] ?? result.predicted_outcome}
          </p>

          {/* Named only when it is not the ensemble. Labelling every answer
              would make this one stop standing out, and it has to: an ELO
              number and an ensemble number are not the same claim. */}
          {result.model && (
            <p className="font-mono text-[11px] text-fg-subtle">
              modelo: {result.model} · o ensemble doméstico não se aplica a confrontos
              entre ligas
            </p>
          )}

          <p className="text-[11px] text-fg-subtle">
            Previsão baseada em dados históricos · Aposta com responsabilidade.
          </p>
        </Panel>
      )}

      {stats?.goal_markets && (
        <Panel className="mt-4">
          <GoalMarketsPanel
            markets={stats.goal_markets}
            homeTeam={stats.home_team}
            awayTeam={stats.away_team}
          />
        </Panel>
      )}

      {stats && (
        <Panel className="mt-4">
          <HeadToHeadPanel h2h={stats.head_to_head} />
        </Panel>
      )}

      {stats && (
        <Panel className="mt-4">
          <TeamComparison home={stats.home} away={stats.away} />
        </Panel>
      )}

      {statsError && !stats && (
        <p className="mt-4 text-xs text-fg-subtle">Estatísticas indisponíveis: {statsError}</p>
      )}

      {loading && (
        <div className="flex items-center gap-2 mt-4 text-xs text-fg-subtle">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />A calcular…
        </div>
      )}
    </PageBody>
  );
}
