import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { RefreshCw, Loader2, Calendar, Download, CircleSlash, Cpu } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { MatchSummaryCard } from '@/components/match/MatchSummaryCard';
import { DetailPanel } from '@/components/layout/DetailPanel';
import { fetchAvailableDates, fetchLeagues, downloadExport, runInference } from '@/lib/api';
import type { InferencePrediction } from '@/lib/api';
import { usePredictions } from '@/contexts/PredictionsContext';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { buildMatchId } from '@/lib/matchId';
import { layout } from '@/config';
import type { League, MatchPrediction } from '@/types';

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
  const navigate = useNavigate();
  const isWide = useMediaQuery(layout.detailPanelQuery);

  const [leagueList, setLeagueList] = useState<League[]>([]);
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [activeLeague, setActiveLeague] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<MatchPrediction | null>(null);
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
    setSelectedMatch(null);
  }, []);

  const handleSelectMatch = useCallback(
    (match: MatchPrediction, leagueCode: string) => {
      if (isWide) {
        setSelectedMatch(match);
      } else {
        const id = buildMatchId(leagueCode, match.home_team, match.away_team);
        navigate(`/match/${id}?date=${encodeURIComponent(selectedDate)}`);
      }
    },
    [isWide, navigate, selectedDate],
  );

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
  const panelOpen = isWide && !!selectedMatch;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-fg">Dashboard</h1>
          <p className="text-sm text-fg-muted mt-1">
            Previsões ML para {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}
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
        <KpiCard label="Ligas" value={leagueList.length} icon="🏆" accent="blue" delay={0} />
        <KpiCard label="Jogos" value={totalMatches} icon="⚽" accent="neutral" delay={0.05} />
        <KpiCard label="Value Bets" value={totalValueBets} icon="💡" accent="green" delay={0.1} />
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
          accent="amber"
          delay={0.15}
        />
      </div>

      {/* Live inference results */}
      {inferError && (
        <GlassCard accent="red" className="mb-4 text-center py-4">
          <p className="text-accent-red text-sm">✗ {inferError}</p>
        </GlassCard>
      )}
      {inferResults && inferResults.length === 0 && (
        <GlassCard className="mb-4 text-center py-4">
          <p className="text-fg-muted text-sm">Sem previsões ao vivo para esta seleção.</p>
        </GlassCard>
      )}
      {inferResults && inferResults.length > 0 && (
        <GlassCard accent="blue" className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-accent-blue" />
            <h3 className="text-sm font-semibold text-fg">Previsões ao vivo (modelo HF)</h3>
            <button
              onClick={() => setInferResults(null)}
              className="ml-auto text-xs text-fg-subtle hover:text-fg"
            >
              Fechar
            </button>
          </div>
          <div className="space-y-2">
            {inferResults.map((r, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b border-line last:border-0">
                <span className="text-sm text-fg-muted">{r.home_team} vs {r.away_team}</span>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-fg-subtle">{r.league ?? ''}</span>
                  <span className="text-accent-blue font-medium">{r.predicted_outcome}</span>
                  <span className="text-fg-subtle">{(r.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Loading / Error */}
      {(loadingLeagues || loading) && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-accent-blue animate-spin" />
          <span className="ml-3 text-fg-muted">
            {loadingLeagues ? 'A carregar ligas...' : 'A carregar previsões...'}
          </span>
        </div>
      )}

      {displayError && !loading && (
        <GlassCard accent="red" className="text-center py-10">
          <p className="text-accent-red">{displayError}</p>
          <NeonButton variant="secondary" size="sm" onClick={() => window.location.reload()} className="mt-4">
            Tentar novamente
          </NeonButton>
        </GlassCard>
      )}

      {/* League tabs + master/detail */}
      {!loadingLeagues && !loading && !displayError && leagueList.length > 0 && (
        <>
          <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-hide pb-1">
            {leagueList.map((league) => (
              <button
                key={league.code}
                onClick={() => {
                  setActiveLeague(league.code);
                  setSelectedMatch(null);
                }}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition-all cursor-pointer border',
                  activeLeague === league.code
                    ? 'bg-accent-blue/12 border-accent-blue/25 text-fg'
                    : 'glass border-line text-fg-muted hover:text-fg',
                )}
              >
                <span>{LEAGUE_ICONS[league.code] || '🏟️'}</span>
                {league.name}
                {leagueDataMap[league.code] && (
                  <span className="text-[10px] text-fg-subtle ml-1">
                    ({leagueDataMap[league.code].matches.length})
                  </span>
                )}
              </button>
            ))}
          </div>

          {currentLeague && currentLeague.matches.length > 0 ? (
            <div className="flex gap-4 items-start">
              <div
                className={clsx(
                  'min-w-0 flex-1 grid gap-3',
                  panelOpen ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2',
                )}
              >
                {currentLeague.matches.map((match, i) => (
                  <MatchSummaryCard
                    key={`${match.home_team}-${match.away_team}`}
                    match={match}
                    index={i}
                    selected={
                      selectedMatch?.home_team === match.home_team &&
                      selectedMatch?.away_team === match.away_team
                    }
                    onSelect={() => handleSelectMatch(match, currentLeague.league_code)}
                  />
                ))}
              </div>

              {isWide && (
                <DetailPanel match={selectedMatch} onClose={() => setSelectedMatch(null)} />
              )}
            </div>
          ) : currentLeague ? (
            <GlassCard className="text-center py-10">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-card-2 flex items-center justify-center">
                <CircleSlash className="w-6 h-6 text-fg-subtle" />
              </div>
              <p className="text-fg-muted">
                Sem jogos agendados para {currentLeague.league_name} em{' '}
                {selectedDate === todayStr() ? 'hoje' : formatDateLabel(selectedDate)}.
              </p>
            </GlassCard>
          ) : null}
        </>
      )}

      {!loadingLeagues && !loading && !displayError && allLeagues.length === 0 && (
        <GlassCard className="text-center py-10">
          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-card-2 flex items-center justify-center">
            <CircleSlash className="w-6 h-6 text-fg-subtle" />
          </div>
          <p className="text-fg-muted">
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
  accent,
  delay,
}: {
  label: string;
  value: number | string;
  icon: string;
  accent: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'neutral';
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
    >
      <GlassCard padding="sm" accent={accent}>
        <div className="flex items-center gap-3">
          <span className="text-2xl">{icon}</span>
          <div>
            <p className="text-xs text-fg-muted">{label}</p>
            <p className="text-xl font-bold text-fg">{value}</p>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}
