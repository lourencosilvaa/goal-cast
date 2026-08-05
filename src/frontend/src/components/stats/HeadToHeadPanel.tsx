import { Swords } from 'lucide-react';
import type { HeadToHead } from '@/lib/api';
import { MatchHistoryList } from './MatchHistoryList';
import { RecordBar, StatSection, StatTile } from './primitives';
import { num, pct } from './format';

/** Head-to-head record and past meetings between the two teams of a fixture. */
export function HeadToHeadPanel({ h2h }: { h2h: HeadToHead }) {
  if (h2h.played === 0) {
    return (
      <StatSection icon={<Swords className="w-3.5 h-3.5" />} title="Confronto Direto">
        <p className="text-xs text-fg-subtle">
          Sem confrontos anteriores nos dados históricos.
        </p>
      </StatSection>
    );
  }

  return (
    <StatSection
      icon={<Swords className="w-3.5 h-3.5" />}
      title="Confronto Direto"
      subtitle={`${h2h.played} ${h2h.played === 1 ? 'jogo' : 'jogos'}`}
    >
      <div className="space-y-4">
        <RecordBar
          wins={h2h.home_wins}
          draws={h2h.draws}
          losses={h2h.away_wins}
          labels={[h2h.home_team, 'Empates', h2h.away_team]}
        />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <StatTile
            label="Golos/jogo"
            value={num(h2h.avg_goals_total, 2)}
            accent="blue"
          />
          <StatTile
            label="Média golos"
            value={`${num(h2h.avg_goals_home, 1)} — ${num(h2h.avg_goals_away, 1)}`}
            hint="casa — fora"
          />
          <StatTile label="Ambas marcam" value={pct(h2h.btts_pct)} />
          <StatTile label="Over 2.5" value={pct(h2h.over_2_5_pct)} />
        </div>

        <div>
          <p className="text-[10px] text-fg-subtle uppercase tracking-wider mb-2">
            Últimos encontros
          </p>
          <MatchHistoryList matches={h2h.matches} subject={h2h.home_team} />
        </div>
      </div>
    </StatSection>
  );
}
