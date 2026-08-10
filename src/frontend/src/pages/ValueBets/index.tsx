import { useEffect, useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { Pill } from '@/components/ui/Pill';
import { PageBody } from '@/components/layout/PageBody';
import { FIELD_CLASS } from '@/components/ui/TeamCombobox';
import { fetchAvailableDates } from '@/lib/api';
import { usePredictions } from '@/contexts/PredictionsContext';
import { dateFilter } from '@/config/theme';
import { describeDate, formatDayPill, todayStr, withToday } from '@/lib/dates';
import type { ValueBet } from '@/types';

interface FlatValueBet extends ValueBet {
  home_team: string;
  away_team: string;
  league: string;
}

const CONFIDENCE_CLASS: Record<string, string> = {
  HIGH: 'text-accent-green border-accent-green/40',
  MEDIUM: 'text-accent-amber border-accent-amber/40',
  LOW: 'text-fg-subtle border-line',
};

export function ValueBetsPage() {
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [availableDates, setAvailableDates] = useState<string[]>([]);

  const { data: allLeagues, loading } = usePredictions(selectedDate);

  useEffect(() => {
    fetchAvailableDates().then(setAvailableDates).catch(() => {});
  }, []);

  const valueBets = useMemo<FlatValueBet[]>(() => {
    const flat: FlatValueBet[] = [];
    for (const league of allLeagues) {
      for (const match of league.matches) {
        for (const bet of match.value_bets) {
          flat.push({
            ...bet,
            home_team: match.home_team,
            away_team: match.away_team,
            league: match.league,
          });
        }
      }
    }
    return flat.sort((a, b) => b.edge - a.edge);
  }, [allLeagues]);

  const displayDates = useMemo(() => withToday(availableDates), [availableDates]);
  const pillDates = displayDates.slice(0, dateFilter.maxPills);
  const overflowDates = displayDates.slice(dateFilter.maxPills);

  return (
    <PageBody intro="Jogos onde a probabilidade do modelo mais diverge da probabilidade implícita nas odds do bookmaker.">
      <div className="flex items-center gap-3 flex-wrap mb-5">
        <div className="flex gap-1.5 flex-wrap">
          {pillDates.map((date) => (
            <Pill
              key={date}
              mono
              active={date === selectedDate}
              onClick={() => setSelectedDate(date)}
              title={date}
            >
              {date === todayStr() ? 'HOJE' : formatDayPill(date)}
            </Pill>
          ))}
          {overflowDates.length > 0 && (
            <select
              value={overflowDates.includes(selectedDate) ? selectedDate : ''}
              onChange={(e) => e.target.value && setSelectedDate(e.target.value)}
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
      </div>

      {loading ? (
        <div className="flex items-center justify-center gap-3 py-16 text-fg-muted text-sm">
          <Loader2 className="w-5 h-5 text-accent animate-spin" />
          A carregar…
        </div>
      ) : valueBets.length === 0 ? (
        <p className="py-10 text-sm text-fg-subtle">
          Sem apostas de valor detetadas para {describeDate(selectedDate)}.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {valueBets.map((bet, i) => (
            <div
              key={`${bet.home_team}-${bet.away_team}-${bet.outcome}-${i}`}
              className="flex items-center gap-4 flex-wrap px-5 py-3.5 rounded-lg border border-line-soft bg-card"
            >
              <span className="w-[120px] shrink-0 font-mono text-[11px] text-fg-subtle truncate">
                {bet.league}
              </span>
              <span className="flex-1 min-w-[180px] text-sm font-semibold text-fg truncate">
                {bet.home_team} vs {bet.away_team}
              </span>
              <span className="w-[120px] shrink-0 text-[13px] text-fg-muted truncate">
                {bet.outcome}
              </span>
              <span className="w-[110px] shrink-0 font-mono text-xs text-fg">
                Modelo {(bet.ml_probability * 100).toFixed(1)}%
              </span>
              <span className="w-[110px] shrink-0 font-mono text-xs text-fg-subtle">
                B365 {(bet.bookmaker_implied * 100).toFixed(1)}%
              </span>
              <span className="w-[70px] shrink-0 font-mono text-xs text-fg-muted">
                @{bet.best_odds.toFixed(2)}
              </span>
              <span className="w-[80px] shrink-0 font-mono text-xs text-fg-subtle">
                Kelly {(bet.kelly_fraction * 100).toFixed(1)}%
              </span>
              <span
                className={clsx(
                  'shrink-0 px-2 py-0.5 rounded border font-mono text-[10px] font-bold',
                  CONFIDENCE_CLASS[bet.confidence] ?? CONFIDENCE_CLASS.LOW,
                )}
              >
                {bet.confidence}
              </span>
              <span className="w-[70px] shrink-0 text-right font-mono text-[13px] font-bold text-accent">
                {bet.edge_pct}
              </span>
            </div>
          ))}
        </div>
      )}
    </PageBody>
  );
}
