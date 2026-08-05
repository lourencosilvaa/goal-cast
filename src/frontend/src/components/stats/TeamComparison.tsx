import { Users } from 'lucide-react';
import type { TeamStats } from '@/lib/api';
import { ComparisonRow, FormPills, StatSection } from './primitives';

/**
 * Side-by-side form comparison of the two teams in a fixture.
 *
 * Everything shown here is empirical (what each side has actually done),
 * complementing the model's probabilities above it.
 */
export function TeamComparison({
  home,
  away,
}: {
  home: TeamStats;
  away: TeamStats;
}) {
  const window = Math.max(home.recent.played, away.recent.played);

  return (
    <StatSection
      icon={<Users className="w-3.5 h-3.5" />}
      title="Comparação de Forma"
      subtitle={window > 0 ? `últimos ${window} jogos` : undefined}
    >
      <div className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1.5">
            <p className="text-xs text-fg-muted truncate">{home.team}</p>
            <FormPills sequence={home.form_sequence} />
          </div>
          <div className="space-y-1.5 flex flex-col items-end">
            <p className="text-xs text-fg-muted truncate">{away.team}</p>
            <FormPills sequence={away.form_sequence} />
          </div>
        </div>

        <div className="space-y-3 pt-1">
          <ComparisonRow
            label="Pontos/jogo"
            homeValue={home.recent.points_per_game}
            awayValue={away.recent.points_per_game}
          />
          <ComparisonRow
            label="Golos marcados"
            homeValue={home.recent.avg_goals_for}
            awayValue={away.recent.avg_goals_for}
          />
          <ComparisonRow
            label="Golos sofridos"
            homeValue={home.recent.avg_goals_against}
            awayValue={away.recent.avg_goals_against}
            highlight="lower"
          />
          <ComparisonRow
            label="Ambas marcam"
            homeValue={home.rates.btts}
            awayValue={away.rates.btts}
            highlight="none"
          />
          <ComparisonRow
            label="Over 2.5"
            homeValue={home.rates.over_2_5}
            awayValue={away.rates.over_2_5}
            highlight="none"
          />
          <ComparisonRow
            label="Remates/jogo"
            homeValue={home.averages.shots}
            awayValue={away.averages.shots}
          />
        </div>

        <div className="grid grid-cols-2 gap-3 pt-1">
          <VenueSummary
            title={`${home.team} em casa`}
            record={home.home}
          />
          <VenueSummary
            title={`${away.team} fora`}
            record={away.away}
          />
        </div>
      </div>
    </StatSection>
  );
}

function VenueSummary({
  title,
  record,
}: {
  title: string;
  record: TeamStats['overall'];
}) {
  return (
    <div className="px-3 py-2 rounded-xl bg-card-2 border border-line">
      <p className="text-[10px] text-fg-subtle truncate mb-1">{title}</p>
      <p className="text-sm font-semibold text-fg">
        {record.wins}-{record.draws}-{record.losses}
      </p>
      <p className="text-[10px] text-fg-subtle mt-0.5">
        {record.played} jogos · {record.points_per_game.toFixed(2)} ppj
      </p>
    </div>
  );
}
