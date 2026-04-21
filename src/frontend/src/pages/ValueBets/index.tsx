import { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Loader2, Calendar } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { fetchAvailableDates, fetchLeagues, fetchLeaguePredictions } from '@/lib/api';
import type { ValueBet } from '@/types';

interface FlatValueBet extends ValueBet {
  home_team: string;
  away_team: string;
  league: string;
}

function confidenceColor(conf: string): string {
  if (conf === 'HIGH') return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/20';
  if (conf === 'MEDIUM') return 'bg-amber-500/15 text-amber-300 border-amber-500/20';
  return 'bg-red-500/15 text-red-300 border-red-500/20';
}

function todayStr(): string {
  const d = new Date();
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  return `${dd}/${mm}/${d.getFullYear()}`;
}

function formatDateLabel(d: string): string {
  const months = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];
  const [day, month] = d.split('/');
  return `${parseInt(day)} ${months[parseInt(month) - 1]}`;
}

export function ValueBetsPage() {
  const [valueBets, setValueBets] = useState<FlatValueBet[]>([]);
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [loading, setLoading] = useState(true);
  // Track which dates have been fetched
  const fetchedRef = useRef<Set<string>>(new Set());

  // Load available dates once
  useEffect(() => {
    (async () => {
      try {
        const dates = await fetchAvailableDates();
        setAvailableDates(dates);
        if (dates.length > 0 && !dates.includes(todayStr())) {
          setSelectedDate(dates[0]);
        }
      } catch {
        // non-fatal — dates dropdown just won't appear
      }
    })();
  }, []);

  // Fetch value bets when date changes
  useEffect(() => {
    if (fetchedRef.current.has(selectedDate)) return;
    fetchedRef.current.add(selectedDate);
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const leagues = await fetchLeagues();
        const results = await Promise.allSettled(
          leagues.map((l) => fetchLeaguePredictions(l.code, selectedDate)),
        );
        const all: FlatValueBet[] = [];
        for (const result of results) {
          if (result.status !== 'fulfilled') continue;
          const league = result.value;
          for (const match of league.matches) {
            for (const vb of match.value_bets) {
              all.push({
                ...vb,
                home_team: match.home_team,
                away_team: match.away_team,
                league: match.league,
              });
            }
          }
        }
        all.sort((a, b) => b.edge - a.edge);
        if (!cancelled) setValueBets(all);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; fetchedRef.current.delete(selectedDate); };
  }, [selectedDate]);

  const handleDateChange = useCallback((date: string) => {
    setSelectedDate(date);
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white/90">Apostas de Valor</h1>
          <p className="text-sm text-white/40 mt-1">
            Oportunidades onde o modelo ML identifica valor vs odds do bookmaker
          </p>
        </div>

        {availableDates.length > 0 && (
          <div className="relative">
            <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-white/40 pointer-events-none" />
            <select
              value={selectedDate}
              onChange={(e) => handleDateChange(e.target.value)}
              className="appearance-none pl-8 pr-8 py-2 rounded-xl text-sm font-medium bg-white/[0.03] border border-white/[0.06] text-white/80 cursor-pointer focus:outline-none focus:border-green-500/30 backdrop-blur-sm"
            >
              {availableDates.map((d) => (
                <option key={d} value={d} className="bg-[#0e0e16] text-white">
                  {d === todayStr() ? `Hoje (${formatDateLabel(d)})` : formatDateLabel(d)}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-green-400 animate-spin" />
        </div>
      ) : valueBets.length === 0 ? (
        <GlassCard className="text-center py-10">
          <TrendingUp className="w-10 h-10 text-white/20 mx-auto mb-3" />
          <p className="text-white/40">
            Sem apostas de valor detetadas para{' '}
            {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}.
          </p>
        </GlassCard>
      ) : (
        <div className="space-y-3">
          {valueBets.map((vb, i) => (
            <motion.div
              key={`${vb.home_team}-${vb.away_team}-${vb.outcome}-${i}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <GlassCard gradient="green" padding="md" className="neon-glow">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-5 h-5 text-green-400" />
                    <div>
                      <p className="text-white/90 font-semibold">
                        {vb.home_team} vs {vb.away_team}
                      </p>
                      <p className="text-xs text-white/40">{vb.league}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="text-center">
                      <p className="text-[10px] text-white/30">Aposta</p>
                      <p className="text-sm font-semibold text-green-300">
                        {vb.outcome}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-white/30">Edge</p>
                      <p className="text-sm font-bold text-emerald-400">
                        {vb.edge_pct}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-white/30">Odds</p>
                      <p className="text-sm font-semibold text-white/80">
                        {vb.best_odds.toFixed(2)}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-white/30">Kelly</p>
                      <p className="text-sm text-white/60">
                        {(vb.kelly_fraction * 100).toFixed(1)}%
                      </p>
                    </div>
                    <span
                      className={`px-2 py-1 rounded-lg text-xs font-medium border ${confidenceColor(vb.confidence)}`}
                    >
                      {vb.confidence}
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex gap-4 text-xs text-white/40">
                  <span>
                    ML: {(vb.ml_probability * 100).toFixed(1)}% vs B365:{' '}
                    {(vb.bookmaker_implied * 100).toFixed(1)}%
                  </span>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
