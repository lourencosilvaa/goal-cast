import { Activity, CalendarDays, Flame, Home, Plane } from 'lucide-react';
import type { TeamStats } from '@/lib/api';
import { MatchHistoryList } from './MatchHistoryList';
import { FormPills, RecordBar, StatSection, StatTile } from './primitives';
import { formTone, num, pct } from './format';

/** Full statistical profile of one team — the body of the "Equipas" page. */
export function TeamStatsPanel({ stats }: { stats: TeamStats }) {
  const { overall, home, away, recent, rates, averages } = stats;
  const recentLabel = `últimos ${recent.played} jogos`;

  return (
    <div className="space-y-6">
      <StatSection
        icon={<Flame className="w-3.5 h-3.5" />}
        title="Forma Recente"
        subtitle={recentLabel}
      >
        <div className="space-y-3">
          <FormPills sequence={stats.form_sequence} />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatTile
              label="Pontos/jogo"
              value={num(recent.points_per_game, 2)}
              accent={formTone(recent.points_per_game)}
            />
            <StatTile
              label="Registo"
              value={`${recent.wins}-${recent.draws}-${recent.losses}`}
              hint="V-E-D"
            />
            <StatTile label="Golos marcados" value={num(recent.avg_goals_for, 2)} />
            <StatTile label="Golos sofridos" value={num(recent.avg_goals_against, 2)} />
          </div>
        </div>
      </StatSection>

      <StatSection
        icon={<Activity className="w-3.5 h-3.5" />}
        title="Registo Histórico"
        subtitle={`${overall.played} jogos`}
      >
        <div className="space-y-4">
          <RecordBar wins={overall.wins} draws={overall.draws} losses={overall.losses} />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatTile label="Pontos/jogo" value={num(overall.points_per_game, 2)} />
            <StatTile label="% Vitórias" value={pct(overall.win_pct, 1)} accent="green" />
            <StatTile
              label="Golos"
              value={`${overall.goals_for}:${overall.goals_against}`}
              hint={`DG ${overall.goal_difference >= 0 ? '+' : ''}${overall.goal_difference}`}
            />
            <StatTile
              label="Média golos"
              value={`${num(overall.avg_goals_for, 2)} / ${num(overall.avg_goals_against, 2)}`}
              hint="marcados / sofridos"
            />
          </div>
        </div>
      </StatSection>

      <div className="grid gap-4 sm:grid-cols-2">
        <VenueCard icon={<Home className="w-3.5 h-3.5" />} title="Em Casa" record={home} />
        <VenueCard icon={<Plane className="w-3.5 h-3.5" />} title="Fora" record={away} />
      </div>

      <StatSection title="Mercados" subtitle={recentLabel}>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <StatTile label="Sem sofrer" value={pct(rates.clean_sheets, 1)} />
          <StatTile label="Não marcou" value={pct(rates.failed_to_score, 1)} />
          <StatTile label="Ambas marcam" value={pct(rates.btts, 1)} accent="blue" />
          <StatTile label="Over 2.5" value={pct(rates.over_2_5, 1)} accent="blue" />
        </div>
      </StatSection>

      <StatSection title="Médias por Jogo" subtitle={recentLabel}>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <StatTile label="Remates" value={num(averages.shots, 1)} />
          <StatTile label="À baliza" value={num(averages.shots_on_target, 1)} />
          <StatTile label="Cantos" value={num(averages.corners, 1)} />
          <StatTile label="Cartões" value={num(averages.cards, 1)} accent="amber" />
        </div>
      </StatSection>

      <StatSection
        icon={<CalendarDays className="w-3.5 h-3.5" />}
        title="Resultados Recentes"
      >
        <MatchHistoryList matches={stats.recent_matches} subject={stats.team} />
      </StatSection>
    </div>
  );
}

function VenueCard({
  icon,
  title,
  record,
}: {
  icon: React.ReactNode;
  title: string;
  record: TeamStats['overall'];
}) {
  return (
    <div className="rounded-lg border border-line bg-card-2/50 p-4 space-y-3">
      <div className="flex items-baseline justify-between">
        <h4 className="text-xs text-fg-subtle uppercase tracking-wider flex items-center gap-1.5">
          {icon}
          {title}
        </h4>
        <span className="text-[10px] text-fg-subtle">{record.played} jogos</span>
      </div>
      <RecordBar wins={record.wins} draws={record.draws} losses={record.losses} />
      <div className="grid grid-cols-2 gap-2">
        <StatTile label="Pontos/jogo" value={num(record.points_per_game, 2)} />
        <StatTile
          label="Golos"
          value={`${num(record.avg_goals_for, 2)} / ${num(record.avg_goals_against, 2)}`}
          hint="marcados / sofridos"
        />
      </div>
    </div>
  );
}
