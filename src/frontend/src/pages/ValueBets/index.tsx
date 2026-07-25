import { useEffect, useState, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Loader2, Calendar, Download } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { fetchAvailableDates, downloadExport } from '@/lib/api';
import { usePredictions } from '@/contexts/PredictionsContext';
import type { ValueBet } from '@/types';

interface FlatValueBet extends ValueBet {
  home_team: string;
  away_team: string;
  league: string;
}

function confidenceColor(conf: string): string {
  if (conf === 'HIGH') return 'bg-accent-green/15 text-accent-green border-accent-green/25';
  if (conf === 'MEDIUM') return 'bg-accent-amber/15 text-accent-amber border-accent-amber/25';
  return 'bg-accent-red/15 text-accent-red border-accent-red/25';
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
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [exporting, setExporting] = useState(false);

  const { data: allLeagues, loading } = usePredictions(selectedDate);

  useEffect(() => {
    fetchAvailableDates()
      .then((dates) => {
        setAvailableDates(dates);
        if (dates.length > 0 && !dates.includes(todayStr())) {
          setSelectedDate(dates[0]);
        }
      })
      .catch(() => {});
  }, []);

  const valueBets = useMemo<FlatValueBet[]>(() => {
    const flat: FlatValueBet[] = [];
    for (const league of allLeagues) {
      for (const match of league.matches) {
        for (const vb of match.value_bets) {
          flat.push({ ...vb, home_team: match.home_team, away_team: match.away_team, league: match.league });
        }
      }
    }
    return flat.sort((a, b) => b.edge - a.edge);
  }, [allLeagues]);

  const handleDateChange = useCallback((date: string) => setSelectedDate(date), []);

  async function handleExport(format: 'csv' | 'excel') {
    setExporting(true);
    try {
      await downloadExport(format, selectedDate);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-fg">Apostas de Valor</h1>
          <p className="text-sm text-fg-muted mt-1">
            Oportunidades onde o modelo ML identifica valor vs odds do bookmaker
          </p>
        </div>

        <div className="flex items-center gap-3">
          {availableDates.length > 0 && (
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-fg-subtle pointer-events-none" />
              <select
                value={selectedDate}
                onChange={(e) => handleDateChange(e.target.value)}
                className="appearance-none pl-8 pr-8 py-2 rounded-xl text-sm font-medium bg-card border border-line text-fg cursor-pointer focus:outline-none focus:border-accent-blue/40"
              >
                {availableDates.map((d) => (
                  <option key={d} value={d} className="bg-card text-fg">
                    {d === todayStr() ? `Hoje (${formatDateLabel(d)})` : formatDateLabel(d)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <NeonButton variant="secondary" size="sm" loading={exporting} onClick={() => handleExport('csv')}>
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            CSV
          </NeonButton>
          <NeonButton variant="secondary" size="sm" loading={exporting} onClick={() => handleExport('excel')}>
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            Excel
          </NeonButton>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-accent-blue animate-spin" />
        </div>
      ) : valueBets.length === 0 ? (
        <GlassCard className="text-center py-10">
          <TrendingUp className="w-10 h-10 text-fg-subtle mx-auto mb-3" />
          <p className="text-fg-muted">
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
              <GlassCard accent="green" padding="md" className="neon-glow">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-5 h-5 text-accent-green" />
                    <div>
                      <p className="text-fg font-semibold">
                        {vb.home_team} vs {vb.away_team}
                      </p>
                      <p className="text-xs text-fg-muted">{vb.league}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="text-center">
                      <p className="text-[10px] text-fg-subtle">Aposta</p>
                      <p className="text-sm font-semibold text-accent-green">{vb.outcome}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-fg-subtle">Edge</p>
                      <p className="text-sm font-bold text-accent-green">{vb.edge_pct}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-fg-subtle">Odds</p>
                      <p className="text-sm font-semibold text-fg">{vb.best_odds.toFixed(2)}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] text-fg-subtle">Kelly</p>
                      <p className="text-sm text-fg-muted">{(vb.kelly_fraction * 100).toFixed(1)}%</p>
                    </div>
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium border ${confidenceColor(vb.confidence)}`}>
                      {vb.confidence}
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex gap-4 text-xs text-fg-muted">
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
