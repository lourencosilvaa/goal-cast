import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Loader2, Calendar, Download, CircleSlash, Cpu } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { MatchCard } from '@/components/match/MatchCard';
import { fetchAvailableDates, fetchLeagues, downloadExport, runInference } from '@/lib/api';
import type { InferencePrediction } from '@/lib/api';
import { usePredictions } from '@/contexts/PredictionsContext';
import type { League } from '@/types';

const LEAGUE_ICONS: Record<string, string> = {
  E0: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  SP1: '🇪🇸',
  D1: '🇩🇪',
  I1: '🇮🇹',
  F1: '🇫🇷',
  P1: '🇵🇹',
  CL: '⭐',
  EL: '🟠',
  ECL: '🔵',
  FA: '🏆',
  CDR: '🥤',
};

function formatDateLabel(d: string): string {
  const months = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];
  const [day, month] = d.split('/');
  return `${parseInt(day)} ${months[parseInt(month) - 1]}`;
}

function todayStr(): string {
  const d = new Date();
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  return `${dd}/${mm}/${d.getFullYear()}`;
}

export function Dashboard() {
  const [leagueList, setLeagueList] = useState<League[]>([]);
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [activeLeague, setActiveLeague] = useState<string | null>(null);
  const [loadingLeagues, setLoadingLeagues] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [inferResults, setInferResults] = useState<InferencePrediction[] | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  const { data: allLeagues, loading, error, invalidate } = usePredictions(selectedDate);

  // Derive per-league map from shared context data
  const leagueDataMap = Object.fromEntries(allLeagues.map((l) => [l.league_code, l]));

  // Load leagues list + available dates once
  useEffect(() => {
    (async () => {
      try {
        const [list, dates] = await Promise.all([fetchLeagues(), fetchAvailableDates()]);
        setLeagueList(list);
        setAvailableDates(dates);
        if (list.length > 0) setActiveLeague(list[0].code);
        if (dates.length > 0 && !dates.includes(todayStr())) {
          setSelectedDate(dates[0]);
        }
      } catch (e) {
        setInitError(e instanceof Error ? e.message : 'Failed to load leagues');
      } finally {
        setLoadingLeagues(false);
      }
    })();
  }, []);

  // When context data arrives, default to first league that has matches
  useEffect(() => {
    if (!activeLeague && allLeagues.length > 0) {
      setActiveLeague(allLeagues[0].league_code);
    }
  }, [allLeagues, activeLeague]);

  const handleDateChange = useCallback((date: string) => {
    setSelectedDate(date);
  }, []);

  async function handleRefresh() {
    setRefreshing(true);
    try {
      await invalidate();
    } finally {
      setRefreshing(false);
    }
  }

  async function handleInference() {
    setInferring(true);
    setInferResults(null);
    setInferError(null);
    try {
      const results = await runInference(selectedDate, activeLeague ?? undefined);
      setInferResults(results);
    } catch (err) {
      setInferError(err instanceof Error ? err.message : 'Erro na inferência');
    } finally {
      setInferring(false);
    }
  }

  async function handleExport(format: 'csv' | 'excel') {
    setExporting(true);
    try {
      await downloadExport(format, selectedDate);
    } finally {
      setExporting(false);
    }
  }

  const currentLeague = activeLeague ? leagueDataMap[activeLeague] : undefined;
  const totalMatches = allLeagues.reduce((s, l) => s + l.matches.length, 0);
  const totalValueBets = allLeagues.reduce(
    (s, l) => s + l.matches.reduce((ms, m) => ms + m.value_bets.length, 0),
    0,
  );

  const displayError = initError || error;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white/90">Dashboard</h1>
          <p className="text-sm text-white/40 mt-1">
            Previsões ML para {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}
          </p>
        </div>

        <div className="flex items-center gap-3">
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

          <NeonButton variant="secondary" size="sm" loading={exporting} onClick={() => handleExport('csv')}>
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            CSV
          </NeonButton>
          <NeonButton variant="secondary" size="sm" loading={exporting} onClick={() => handleExport('excel')}>
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            Excel
          </NeonButton>
          <NeonButton variant="secondary" size="sm" loading={refreshing} onClick={handleRefresh}>
            <RefreshCw className="w-3.5 h-3.5 mr-1.5 inline" />
            Atualizar
          </NeonButton>
          <NeonButton variant="secondary" size="sm" loading={inferring} onClick={handleInference}>
            <Cpu className="w-3.5 h-3.5 mr-1.5 inline" />
            Calcular ao vivo
          </NeonButton>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        <KpiCard label="Ligas" value={leagueList.length} icon="🏆" delay={0} />
        <KpiCard label="Jogos" value={totalMatches} icon="⚽" delay={0.05} />
        <KpiCard label="Value Bets" value={totalValueBets} icon="💡" delay={0.1} />
        <KpiCard
          label="Melhor Edge"
          value={
            totalValueBets > 0
              ? Math.max(
                  ...allLeagues.flatMap((l) =>
                    l.matches.flatMap((m) => m.value_bets.map((v) => v.edge)),
                  ),
                )
                  .toFixed(1)
                  .toString() + '%'
              : '—'
          }
          icon="🎯"
          delay={0.15}
        />
      </div>

      {/* Live inference results */}
      {inferError && (
        <GlassCard className="mb-4 text-center py-4">
          <p className="text-red-400 text-sm">✗ {inferError}</p>
        </GlassCard>
      )}
      {inferResults && inferResults.length === 0 && (
        <GlassCard className="mb-4 text-center py-4">
          <p className="text-white/40 text-sm">Sem previsões ao vivo para esta seleção.</p>
        </GlassCard>
      )}
      {inferResults && inferResults.length > 0 && (
        <GlassCard className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-green-400" />
            <h3 className="text-sm font-semibold text-white/80">Previsões ao vivo (modelo HF)</h3>
            <button
              onClick={() => setInferResults(null)}
              className="ml-auto text-xs text-white/30 hover:text-white/60"
            >
              Fechar
            </button>
          </div>
          <div className="space-y-2">
            {inferResults.map((r, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                <span className="text-sm text-white/70">{r.home_team} vs {r.away_team}</span>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-white/50">{r.league ?? ''}</span>
                  <span className="text-green-400 font-medium">{r.predicted_outcome}</span>
                  <span className="text-white/30">{(r.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Loading / Error */}
      {(loadingLeagues || loading) && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-green-400 animate-spin" />
          <span className="ml-3 text-white/40">
            {loadingLeagues ? 'A carregar ligas...' : 'A carregar previsões...'}
          </span>
        </div>
      )}

      {displayError && !loading && (
        <GlassCard className="text-center py-10">
          <p className="text-red-400">{displayError}</p>
          <NeonButton variant="secondary" size="sm" onClick={() => window.location.reload()} className="mt-4">
            Tentar novamente
          </NeonButton>
        </GlassCard>
      )}

      {/* League tabs */}
      {!loadingLeagues && !loading && !displayError && leagueList.length > 0 && (
        <>
          <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-hide pb-1">
            {leagueList.map((league) => (
              <button
                key={league.code}
                onClick={() => setActiveLeague(league.code)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition-all cursor-pointer ${
                  activeLeague === league.code
                    ? 'bg-gradient-to-r from-green-600/20 to-emerald-600/10 border border-green-500/20 text-white'
                    : 'glass border border-white/[0.06] text-white/50 hover:text-white/80'
                }`}
              >
                <span>{LEAGUE_ICONS[league.code] || '🏟️'}</span>
                {league.name}
                {leagueDataMap[league.code] && (
                  <span className="text-[10px] text-white/30 ml-1">
                    ({leagueDataMap[league.code].matches.length})
                  </span>
                )}
              </button>
            ))}
          </div>

          {currentLeague && currentLeague.matches.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {currentLeague.matches.map((match, i) => (
                <MatchCard
                  key={`${match.home_team}-${match.away_team}`}
                  match={match}
                  index={i}
                />
              ))}
            </div>
          ) : currentLeague ? (
            <GlassCard className="text-center py-10">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-white/5 flex items-center justify-center">
                <CircleSlash className="w-6 h-6 text-white/20" />
              </div>
              <p className="text-white/40">
                Sem jogos agendados para {currentLeague.league_name} em{' '}
                {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}.
              </p>
            </GlassCard>
          ) : null}
        </>
      )}

      {!loadingLeagues && !loading && !displayError && allLeagues.length === 0 && (
        <GlassCard className="text-center py-10">
          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-white/5 flex items-center justify-center">
            <CircleSlash className="w-6 h-6 text-white/20" />
          </div>
          <p className="text-white/40">
            Sem jogos agendados para{' '}
            {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}.
          </p>
        </GlassCard>
      )}
    </div>
  );
}

function KpiCard({
  label,
  value,
  icon,
  delay,
}: {
  label: string;
  value: number | string;
  icon: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
    >
      <GlassCard padding="sm">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{icon}</span>
          <div>
            <p className="text-xs text-white/40">{label}</p>
            <p className="text-xl font-bold text-white/90">{value}</p>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}
