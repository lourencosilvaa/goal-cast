import { useCallback, useState } from 'react';
import { Loader2, Search } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { LeagueSelect } from '@/components/ui/LeagueSelect';
import { TeamCombobox } from '@/components/ui/TeamCombobox';
import { PageBody } from '@/components/layout/PageBody';
import { TeamStatsPanel } from '@/components/stats/TeamStatsPanel';
import { useLeagues } from '@/hooks/useLeagues';
import { useTeams } from '@/hooks/useTeams';
import { fetchTeamStats, type TeamStats } from '@/lib/api';

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="p-4 rounded-lg border border-line-soft bg-card">
      <div className="font-mono text-2xl font-bold text-accent leading-none">{value}</div>
      <div className="text-[11px] text-fg-subtle mt-1.5">{label}</div>
    </div>
  );
}

/** Statistical profile of any single team in a supported league. */
export function TeamStatsPage() {
  const [leagueCode, setLeagueCode] = useState('');
  const [team, setTeam] = useState('');
  const [stats, setStats] = useState<TeamStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { domestic } = useLeagues();
  const { teamsFor, error: teamsError } = useTeams();

  /*
   * Derived rather than stored: the default is the first league the backend
   * offers, which is not known at mount. Deriving avoids an effect that would
   * write state as soon as the list arrives.
   */
  const activeLeague = leagueCode || domestic[0]?.code || '';
  const currentTeams = teamsFor(activeLeague);

  function handleLeagueChange(code: string) {
    setLeagueCode(code);
    setTeam('');
    setStats(null);
    setError(null);
  }

  const load = useCallback(async (name: string, code: string) => {
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
  }, []);

  function handleTeamChange(name: string) {
    setTeam(name);
    if (name && activeLeague) void load(name, activeLeague);
  }

  return (
    <PageBody
      maxWidth="md"
      intro="Escolhe uma equipa para ver o seu registo histórico, forma recente e médias."
    >
      <Panel overflow="visible" className="flex flex-col gap-4">
        <LeagueSelect value={activeLeague} onChange={handleLeagueChange} />
        <TeamCombobox
          label="Equipa"
          value={team}
          onChange={handleTeamChange}
          teams={currentTeams}
          placeholder="Selecionar equipa…"
        />
        {teamsError && <p className="text-xs text-accent-red">{teamsError}</p>}

        <Button
          onClick={() => team.trim() && activeLeague && void load(team.trim(), activeLeague)}
          loading={loading}
          disabled={!team.trim() || !activeLeague}
          className="w-full"
        >
          <Search className="w-4 h-4" />
          Ver estatísticas
        </Button>
      </Panel>

      {error && (
        <Panel accent="red" padding="sm" className="mt-4">
          <p className="text-sm text-accent-red">{error}</p>
        </Panel>
      )}

      {loading && (
        <div className="flex items-center gap-2 mt-4 text-xs text-fg-subtle">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />A carregar…
        </div>
      )}

      {!stats && !error && !loading && (
        <p className="mt-6 text-sm text-fg-subtle">
          Nenhuma equipa selecionada.
        </p>
      )}

      {stats && (
        <>
          <div className="flex items-baseline gap-3 mt-6 mb-4">
            <span className="text-xl font-extrabold text-fg">{stats.team}</span>
            <SectionLabel>{stats.league}</SectionLabel>
            <span className="ml-auto font-mono text-sm font-bold text-fg">
              {stats.overall.wins}-{stats.overall.draws}-{stats.overall.losses}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
            <StatCard value={String(stats.overall.played)} label="Jogos registados" />
            <StatCard
              value={stats.overall.points_per_game.toFixed(2)}
              label="Pontos por jogo"
            />
            <StatCard
              value={stats.overall.avg_goals_for.toFixed(2)}
              label="Golos marcados / jogo"
            />
            <StatCard
              value={stats.overall.avg_goals_against.toFixed(2)}
              label="Golos sofridos / jogo"
            />
          </div>

          <Panel>
            <TeamStatsPanel stats={stats} />
          </Panel>
        </>
      )}
    </PageBody>
  );
}
