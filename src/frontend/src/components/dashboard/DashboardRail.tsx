import { SectionLabel } from '@/components/ui/SectionLabel';
import { rail } from '@/config/theme';
import type { DashboardMetrics } from './derive';

/**
 * Standing statistics column beside the fixture list.
 *
 * Everything here is derived from the same payload the list renders — there is
 * no separate metrics endpoint, and inventing one client-side would put
 * numbers on screen the backend could not reproduce.
 */
export function DashboardRail({ metrics }: { metrics: DashboardMetrics }) {
  const tallest = Math.max(1, ...metrics.perLeague.map((entry) => entry.count));

  return (
    <div
      className="hidden lg:flex flex-col gap-6 shrink-0 border-r border-line p-5 overflow-y-auto"
      style={{ width: rail.dashboardWidthPx }}
    >
      <div>
        <SectionLabel className="mb-2.5">Jogos</SectionLabel>
        <div className="flex items-baseline gap-2 mb-2.5">
          <span className="font-mono text-[34px] font-bold text-accent leading-none">
            {metrics.matchCount}
          </span>
          <span className="text-xs text-fg-subtle">
            em {metrics.leagueCount} liga{metrics.leagueCount === 1 ? '' : 's'}
          </span>
        </div>

        {metrics.perLeague.length > 0 && (
          <div className="flex gap-[3px] items-end h-6">
            {metrics.perLeague.map((entry) => (
              <div
                key={entry.code}
                className="w-2 rounded-[1px] bg-accent"
                style={{ height: `${Math.max(20, (entry.count / tallest) * 100)}%` }}
                title={`${entry.name}: ${entry.count} jogo(s)`}
              />
            ))}
          </div>
        )}
      </div>

      <div>
        <SectionLabel className="mb-2.5">Value bets</SectionLabel>
        <span className="font-mono text-[28px] font-bold text-fg leading-none">
          {metrics.valueBetCount}
        </span>
      </div>

      <div>
        <SectionLabel className="mb-2.5">Melhor edge</SectionLabel>
        <span className="font-mono text-[28px] font-bold text-accent-green leading-none">
          {metrics.bestEdge !== null ? `${metrics.bestEdge.toFixed(1)}%` : '—'}
        </span>
      </div>
    </div>
  );
}
