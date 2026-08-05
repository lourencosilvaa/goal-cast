import { BarChart3, Target } from 'lucide-react';
import type { GoalMarkets } from '@/lib/api';
import { StatSection, StatTile } from './primitives';
import { num, pct } from './format';

/**
 * Score-model markets for a fixture: expected goals, over/under lines, both
 * teams to score and the most likely scorelines.
 */
export function GoalMarketsPanel({
  markets,
  homeTeam,
  awayTeam,
}: {
  markets: GoalMarkets;
  homeTeam: string;
  awayTeam: string;
}) {
  const provenance =
    markets.source === 'model'
      ? 'modelo Poisson calibrado'
      : 'estimativa por médias históricas';

  return (
    <div className="space-y-5">
      <StatSection
        icon={<Target className="w-3.5 h-3.5" />}
        title="Golos Esperados (xG)"
        subtitle={provenance}
      >
        <div className="grid grid-cols-3 gap-2">
          <StatTile label={homeTeam} value={num(markets.expected_goals.home, 2)} />
          <StatTile
            label="Total"
            value={num(markets.expected_goals.total, 2)}
            accent="blue"
          />
          <StatTile label={awayTeam} value={num(markets.expected_goals.away, 2)} />
        </div>
      </StatSection>

      <StatSection
        icon={<BarChart3 className="w-3.5 h-3.5" />}
        title="Mercado de Golos"
      >
        <div className="grid grid-cols-4 gap-2">
          <StatTile label="Over 1.5" value={pct(markets.over_under.over_1_5, 1)} accent="blue" />
          <StatTile label="Over 2.5" value={pct(markets.over_under.over_2_5, 1)} accent="blue" />
          <StatTile label="Over 3.5" value={pct(markets.over_under.over_3_5, 1)} accent="blue" />
          <StatTile label="Under 2.5" value={pct(markets.over_under.under_2_5, 1)} accent="blue" />
        </div>
      </StatSection>

      <StatSection title="Ambas Marcam">
        <div className="grid grid-cols-2 gap-2">
          <StatTile
            label="Sim"
            value={pct(markets.btts.yes, 1)}
            accent={markets.btts.yes > 0.5 ? 'green' : undefined}
          />
          <StatTile
            label="Não"
            value={pct(markets.btts.no, 1)}
            accent={markets.btts.no > 0.5 ? 'green' : undefined}
          />
        </div>
      </StatSection>

      {markets.top_scorelines.length > 0 && (
        <StatSection title="Resultados Mais Prováveis">
          <div className="flex gap-2 flex-wrap">
            {markets.top_scorelines.map((scoreline, i) => (
              <span
                key={scoreline.score}
                className={`text-xs px-2.5 py-1 rounded-lg border ${
                  i === 0
                    ? 'bg-accent-green/15 text-accent-green border-accent-green/25'
                    : 'bg-card-2 text-fg-muted border-line'
                }`}
              >
                {scoreline.score} ({pct(scoreline.prob, 1)})
              </span>
            ))}
          </div>
        </StatSection>
      )}
    </div>
  );
}
