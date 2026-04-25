import { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Loader2, Trophy, Calendar, Download } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { MatchCard } from '@/components/match/MatchCard';
import {
  fetchAvailableDates,
  fetchLeagues,
  fetchLeaguePredictions,
  refreshPredictions,
  downloadExport,
} from '@/lib/api';
import type { League, LeaguePredictions } from '@/types';

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

/** Format DD/MM/YYYY → readable label like "19 Abr" */
function formatDateLabel(d: string): string {
  const months = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];
  const [day, month] = d.split('/');
  return `${parseInt(day)} ${months[parseInt(month) - 1]}`;
}

/** Today in DD/MM/YYYY */
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
  const [leagueData, setLeagueData] = useState<
    Record<string, LeaguePredictions>
  >({});
  const [activeLeague, setActiveLeague] = useState<string | null>(null);
  const [loadingLeagues, setLoadingLeagues] = useState(true);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track which league+date combos have been fetched
  const fetchedRef = useRef<Set<string>>(new Set());

  // Load leagues + available dates (lightweight, once)
  useEffect(() => {
    (async () => {
      try {
        const [list, dates] = await Promise.all([
          fetchLeagues(),
          fetchAvailableDates(),
        ]);
        setLeagueList(list);
        setAvailableDates(dates);
        if (list.length > 0) setActiveLeague(list[0].code);
        // If today isn't in the available dates, select the first one
        if (dates.length > 0 && !dates.includes(todayStr())) {
          setSelectedDate(dates[0]);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load leagues');
      } finally {
        setLoadingLeagues(false);
      }
    })();
  }, []);

  // Load predictions when the active league or date changes
  useEffect(() => {
    if (!activeLeague) return;
    const cacheKey = `${activeLeague}:${selectedDate}`;
    if (fetchedRef.current.has(cacheKey)) return;
    fetchedRef.current.add(cacheKey);
    let cancelled = false;
    (async () => {
      setLoadingPredictions(true);
      setError(null);
      try {
        const data = await fetchLeaguePredictions(activeLeague, selectedDate);
        if (!cancelled) {
          setLeagueData((prev) => ({ ...prev, [activeLeague]: data }));
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load');
          fetchedRef.current.delete(cacheKey);
        }
      } finally {
        if (!cancelled) setLoadingPredictions(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeLeague, selectedDate]);

  const handleDateChange = useCallback((date: string) => {
    setSelectedDate(date);
    // Clear both local state and fetch guard so leagues re-fetch for the new date
    setLeagueData({});
    fetchedRef.current.clear();
  }, []);

  async function handleExport(format: 'csv' | 'excel') {
    setExporting(true);
    try {
      await downloadExport(format, selectedDate);
    } finally {
      setExporting(false);
    }
  }

  async function handleRefresh() {
    if (!activeLeague) return;
    setRefreshing(true);
    try {
      await refreshPredictions(activeLeague);
      const cacheKey = `${activeLeague}:${selectedDate}`;
      fetchedRef.current.delete(cacheKey);
      setLeagueData((prev) => {
        const next = { ...prev };
        delete next[activeLeague];
        return next;
      });
      // Re-fetch
      const data = await fetchLeaguePredictions(activeLeague, selectedDate);
      fetchedRef.current.add(cacheKey);
      setLeagueData((prev) => ({ ...prev, [activeLeague]: data }));
    } finally {
      setRefreshing(false);
    }
  }

  const currentLeague = activeLeague ? leagueData[activeLeague] : undefined;
  const allLoaded = Object.values(leagueData);
  const totalMatches = allLoaded.reduce((s, l) => s + l.matches.length, 0);
  const totalValueBets = allLoaded.reduce(
    (s, l) => s + l.matches.reduce((ms, m) => ms + m.value_bets.length, 0),
    0,
  );

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
          {/* Date selector */}
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

          <NeonButton
            variant="secondary"
            size="sm"
            loading={exporting}
            onClick={() => handleExport('csv')}
          >
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            CSV
          </NeonButton>
          <NeonButton
            variant="secondary"
            size="sm"
            loading={exporting}
            onClick={() => handleExport('excel')}
          >
            <Download className="w-3.5 h-3.5 mr-1.5 inline" />
            Excel
          </NeonButton>
          <NeonButton
            variant="secondary"
            size="sm"
            loading={refreshing}
            onClick={handleRefresh}
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1.5 inline" />
            Atualizar
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
                  ...allLoaded.flatMap((l) =>
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

      {/* Loading / Error */}
      {loadingLeagues && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-green-400 animate-spin" />
          <span className="ml-3 text-white/40">A carregar ligas...</span>
        </div>
      )}

      {error && (
        <GlassCard className="text-center py-10">
          <p className="text-red-400">{error}</p>
          <NeonButton variant="secondary" size="sm" onClick={() => window.location.reload()} className="mt-4">
            Tentar novamente
          </NeonButton>
        </GlassCard>
      )}

      {/* League tabs */}
      {!loadingLeagues && !error && leagueList.length > 0 && (
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
                {leagueData[league.code] && (
                  <span className="text-[10px] text-white/30 ml-1">
                    ({leagueData[league.code].matches.length})
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Match cards */}
          {loadingPredictions && (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 text-green-400 animate-spin" />
              <span className="ml-3 text-white/40">A carregar previsões...</span>
            </div>
          )}

          {!loadingPredictions && currentLeague && currentLeague.matches.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {currentLeague.matches.map((match, i) => (
                <MatchCard
                  key={`${match.home_team}-${match.away_team}`}
                  match={match}
                  index={i}
                />
              ))}
            </div>
          ) : !loadingPredictions && currentLeague ? (
            <GlassCard className="text-center py-10">
              <Trophy className="w-10 h-10 text-white/20 mx-auto mb-3" />
              <p className="text-white/40">
                Sem jogos agendados para {currentLeague.league_name} em{' '}
                {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}.
              </p>
            </GlassCard>
          ) : null}
        </>
      )}

      {!loadingLeagues && !error && leagueList.length === 0 && (
        <GlassCard className="text-center py-10">
          <Trophy className="w-10 h-10 text-white/20 mx-auto mb-3" />
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
