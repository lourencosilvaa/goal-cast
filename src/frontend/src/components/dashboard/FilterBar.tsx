import type { ReactNode } from 'react';
import { Pill } from '@/components/ui/Pill';
import { FIELD_CLASS } from '@/components/ui/TeamCombobox';
import { dateFilter } from '@/config/theme';
import { describeDate, formatDayPill, todayStr } from '@/lib/dates';

export interface LeagueFilterOption {
  code: string;
  name: string;
  /** Fixtures on the selected date; shown so an empty league reads as empty. */
  count: number;
}

interface FilterBarProps {
  dates: string[];
  selectedDate: string;
  onSelectDate: (date: string) => void;
  leagues: LeagueFilterOption[];
  activeLeagues: string[];
  onToggleLeague: (code: string) => void;
  /** Export / refresh controls, pushed to the trailing edge. */
  actions?: ReactNode;
}

/**
 * Date and league filters for the dashboard.
 *
 * Leagues are multi-select rather than tabs: the API returns every league for
 * a date in one payload, and the list below is grouped by league anyway, so
 * narrowing is a filter over one list instead of a switch between several.
 */
export function FilterBar({
  dates,
  selectedDate,
  onSelectDate,
  leagues,
  activeLeagues,
  onToggleLeague,
  actions,
}: FilterBarProps) {
  const pillDates = dates.slice(0, dateFilter.maxPills);
  const overflowDates = dates.slice(dateFilter.maxPills);
  // A date chosen from the overflow select must stay visible somewhere.
  const selectedIsOverflow = overflowDates.includes(selectedDate);

  return (
    <div className="flex items-center gap-4 flex-wrap px-5 md:px-7 py-3.5 border-b border-line shrink-0">
      <div className="flex gap-1.5 flex-wrap">
        {pillDates.map((date) => (
          <Pill
            key={date}
            mono
            active={date === selectedDate}
            onClick={() => onSelectDate(date)}
            title={date}
          >
            {date === todayStr() ? 'HOJE' : formatDayPill(date)}
          </Pill>
        ))}

        {overflowDates.length > 0 && (
          <select
            value={selectedIsOverflow ? selectedDate : ''}
            onChange={(e) => e.target.value && onSelectDate(e.target.value)}
            aria-label="Outras datas"
            className={`${FIELD_CLASS} py-1.5 text-xs font-mono cursor-pointer`}
          >
            <option value="">Mais datas…</option>
            {overflowDates.map((date) => (
              <option key={date} value={date} className="bg-card text-fg">
                {describeDate(date)}
              </option>
            ))}
          </select>
        )}
      </div>

      {leagues.length > 0 && (
        <div className="flex gap-1.5 flex-wrap">
          {leagues.map((league) => (
            <Pill
              key={league.code}
              active={activeLeagues.includes(league.code)}
              onClick={() => onToggleLeague(league.code)}
              title={`${league.name} — ${league.count} jogo(s)`}
            >
              {league.name}
              <span className="ml-1.5 font-mono text-[10px] text-fg-subtle">{league.count}</span>
            </Pill>
          ))}
        </div>
      )}

      {actions && <div className="flex gap-2 ml-auto flex-wrap">{actions}</div>}
    </div>
  );
}
