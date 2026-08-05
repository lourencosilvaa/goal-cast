import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Wand2, Loader2 } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { LeagueSelect } from '@/components/ui/LeagueSelect';
import { TeamCombobox } from '@/components/ui/TeamCombobox';
import { HeadToHeadPanel } from '@/components/stats/HeadToHeadPanel';
import { GoalMarketsPanel } from '@/components/stats/GoalMarketsPanel';
import { TeamComparison } from '@/components/stats/TeamComparison';
import { DEFAULT_LEAGUE_CODE } from '@/config/leagues';
import {
  fetchMatchStats,
  fetchTeams,
  predictCustom,
  type CustomPrediction,
  type MatchStats,
} from '@/lib/api';

const OUTCOME_LABELS: Record<string, string> = {
  'Home Win': 'Vitória Casa',
  Draw: 'Empate',
  'Away Win': 'Vitória Fora',
};

const OUTCOME_COLORS: Record<string, string> = {
  'Home Win': 'text-accent-green',
  Draw: 'text-accent-amber',
  'Away Win': 'text-accent-blue',
};

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-fg-muted">{label}</span>
        <span className={`font-semibold ${color}`}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-card-2 overflow-hidden">
        <motion.div
          className={`h-full rounded-full bg-current ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

export function CustomPredictPage() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [leagueCode, setLeagueCode] = useState(DEFAULT_LEAGUE_CODE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CustomPrediction | null>(null);
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [teamsByLeague, setTeamsByLeague] = useState<Record<string, string[]>>({});

  useEffect(() => {
    fetchTeams()
      .then(setTeamsByLeague)
      .catch(() => {});
  }, []);

  const currentTeams = teamsByLeague[leagueCode] ?? [];

  function handleLeagueChange(code: string) {
    setLeagueCode(code);
    setHomeTeam('');
    setAwayTeam('');
    setResult(null);
    setStats(null);
    setStatsError(null);
    setError(null);
  }

  async function handlePredict() {
    const home = homeTeam.trim();
    const away = awayTeam.trim();
    if (!home || !away) return;
    setLoading(true);
    setResult(null);
    setStats(null);
    setStatsError(null);
    setError(null);

    // The prediction and the statistics are independent reads: a statistics
    // failure must never hide a successful prediction.
    const [prediction, matchStats] = await Promise.allSettled([
      predictCustom(home, away, leagueCode),
      fetchMatchStats(home, away, leagueCode),
    ]);

    if (prediction.status === 'fulfilled') {
      setResult(prediction.value);
    } else {
      setError(
        prediction.reason instanceof Error
          ? prediction.reason.message
          : 'Erro ao obter previsão',
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
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-fg">Previsão Personalizada</h1>
        <p className="text-sm text-fg-muted mt-1">
          Escolhe duas equipas e obtém a previsão do modelo com o histórico do confronto
        </p>
      </div>

      <GlassCard className="space-y-5">
        <LeagueSelect value={leagueCode} onChange={handleLeagueChange} />

        <div className="grid grid-cols-2 gap-4">
          <TeamCombobox
            label="Equipa Casa"
            value={homeTeam}
            onChange={setHomeTeam}
            teams={currentTeams.filter((t) => t !== awayTeam)}
            placeholder="Selecionar..."
          />
          <TeamCombobox
            label="Equipa Fora"
            value={awayTeam}
            onChange={setAwayTeam}
            teams={currentTeams.filter((t) => t !== homeTeam)}
            placeholder="Selecionar..."
          />
        </div>

        <NeonButton
          variant="primary"
          size="md"
          loading={loading}
          onClick={handlePredict}
          disabled={!homeTeam.trim() || !awayTeam.trim()}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
              A calcular...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4 mr-2 inline" />
              Prever
            </>
          )}
        </NeonButton>
      </GlassCard>

      {error && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-4">
          <GlassCard className="text-center py-4">
            <p className="text-accent-red text-sm">✗ {error}</p>
          </GlassCard>
        </motion.div>
      )}

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6"
        >
          <GlassCard className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-fg-subtle mb-1">{result.league}</p>
                <p className="text-lg font-bold text-fg">
                  {result.home_team} <span className="text-fg-subtle font-normal">vs</span>{' '}
                  {result.away_team}
                </p>
              </div>
              <div className="text-right">
                <p
                  className={`text-xl font-bold ${
                    OUTCOME_COLORS[result.predicted_outcome] ?? 'text-fg'
                  }`}
                >
                  {OUTCOME_LABELS[result.predicted_outcome] ?? result.predicted_outcome}
                </p>
                <p className="text-xs text-fg-subtle mt-0.5">
                  {(result.confidence * 100).toFixed(0)}% confiança
                </p>
              </div>
            </div>

            <div className="space-y-3 pt-2 border-t border-line">
              <ProbBar
                label="Vitória Casa"
                value={result.probabilities.home_win}
                color="text-accent-green"
              />
              <ProbBar label="Empate" value={result.probabilities.draw} color="text-accent-amber" />
              <ProbBar
                label="Vitória Fora"
                value={result.probabilities.away_win}
                color="text-accent-blue"
              />
            </div>

            <p className="text-[10px] text-fg-subtle text-center">
              Previsão baseada em dados históricos · Aposte com responsabilidade
            </p>
          </GlassCard>
        </motion.div>
      )}

      {stats?.goal_markets && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="mt-4"
        >
          <GlassCard>
            <GoalMarketsPanel
              markets={stats.goal_markets}
              homeTeam={stats.home_team}
              awayTeam={stats.away_team}
            />
          </GlassCard>
        </motion.div>
      )}

      {stats && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mt-4"
        >
          <GlassCard>
            <HeadToHeadPanel h2h={stats.head_to_head} />
          </GlassCard>
        </motion.div>
      )}

      {stats && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mt-4"
        >
          <GlassCard>
            <TeamComparison home={stats.home} away={stats.away} />
          </GlassCard>
        </motion.div>
      )}

      {statsError && !stats && (
        <div className="mt-4">
          <GlassCard className="text-center py-3">
            <p className="text-xs text-fg-subtle">Estatísticas indisponíveis: {statsError}</p>
          </GlassCard>
        </div>
      )}
    </div>
  );
}
