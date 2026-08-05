import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Search, Shield } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { LeagueSelect } from '@/components/ui/LeagueSelect';
import { TeamCombobox } from '@/components/ui/TeamCombobox';
import { TeamStatsPanel } from '@/components/stats/TeamStatsPanel';
import { DEFAULT_LEAGUE_CODE } from '@/config/leagues';
import { fetchTeamStats, fetchTeams, type TeamStats } from '@/lib/api';

/** Statistical profile of any single team in a supported league. */
export function TeamStatsPage() {
  const [leagueCode, setLeagueCode] = useState(DEFAULT_LEAGUE_CODE);
  const [team, setTeam] = useState('');
  const [stats, setStats] = useState<TeamStats | null>(null);
  const [loading, setLoading] = useState(false);
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
    setTeam('');
    setStats(null);
    setError(null);
  }

  const load = useCallback(
    async (name: string, code: string) => {
      setLoading(true);
      setError(null);
      setStats(null);
      try {
        setStats(await fetchTeamStats(name, code));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erro ao obter estatísticas');
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  function handleTeamChange(name: string) {
    setTeam(name);
    if (name) void load(name, leagueCode);
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-fg">Estatísticas de Equipa</h1>
        <p className="text-sm text-fg-muted mt-1">
          Registo histórico, forma recente e médias de qualquer equipa
        </p>
      </div>

      <GlassCard className="space-y-5">
        <LeagueSelect value={leagueCode} onChange={handleLeagueChange} />
        <TeamCombobox
          label="Equipa"
          value={team}
          onChange={handleTeamChange}
          teams={currentTeams}
          placeholder="Selecionar equipa..."
        />
        <NeonButton
          variant="primary"
          size="md"
          loading={loading}
          onClick={() => team.trim() && void load(team.trim(), leagueCode)}
          disabled={!team.trim()}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
              A carregar...
            </>
          ) : (
            <>
              <Search className="w-4 h-4 mr-2 inline" />
              Ver estatísticas
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

      {!stats && !error && !loading && (
        <div className="mt-6 text-center text-fg-subtle">
          <Shield className="w-8 h-8 mx-auto mb-2 opacity-40" />
          <p className="text-sm">Escolhe uma equipa para ver o seu perfil estatístico</p>
        </div>
      )}

      {stats && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6"
        >
          <GlassCard className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-fg-subtle mb-1">{stats.league}</p>
                <p className="text-lg font-bold text-fg">{stats.team}</p>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold text-fg">
                  {stats.overall.wins}-{stats.overall.draws}-{stats.overall.losses}
                </p>
                <p className="text-xs text-fg-subtle mt-0.5">V-E-D em {stats.overall.played} jogos</p>
              </div>
            </div>
            <div className="pt-2 border-t border-line">
              <TeamStatsPanel stats={stats} />
            </div>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
