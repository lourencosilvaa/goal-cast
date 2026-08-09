import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Cpu, Download, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { DetailPanel } from '@/components/layout/DetailPanel';
import { FilterBar, type LeagueFilterOption } from '@/components/dashboard/FilterBar';
import { DashboardRail } from '@/components/dashboard/DashboardRail';
import { MatchOfTheDay } from '@/components/dashboard/MatchOfTheDay';
import { MatchRow } from '@/components/dashboard/MatchRow';
import { SignalTicker } from '@/components/dashboard/SignalTicker';
import { buildSignals, deriveMetrics, pickHeroMatch } from '@/components/dashboard/derive';
import { downloadExport, fetchAvailableDates, runInference } from '@/lib/api';
import type { InferencePrediction } from '@/lib/api';
import { usePredictions } from '@/contexts/PredictionsContext';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { buildMatchId } from '@/lib/matchId';
import { describeDate, todayStr, withToday } from '@/lib/dates';
import { layout } from '@/config';
import type { MatchPrediction } from '@/types';

export function Dashboard() {
  const navigate = useNavigate();
  const isWide = useMediaQuery(layout.detailPanelQuery);

  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(todayStr());
  const [mutedLeagues, setMutedLeagues] = useState<string[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<MatchPrediction | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [inferResults, setInferResults] = useState<InferencePrediction[] | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);

  const { data: allLeagues, loading, error, invalidate } = usePredictions(selectedDate);

  useEffect(() => {
    fetchAvailableDates().then(setAvailableDates).catch(() => {});
  }, []);

  /*
   * Leagues are tracked as an opt-out set rather than an opt-in one: the day's
   * league list is only known once the payload arrives, and an opt-in default
   * would flash an empty dashboard on every date change.
   */
  const leagueOptions = useMemo<LeagueFilterOption[]>(
    () =>
      allLeagues.map((league) => ({
        code: league.league_code,
        name: league.league_name,
        count: league.matches.length,
      })),
    [allLeagues],
  );

  const activeLeagues = useMemo(
    () => leagueOptions.map((l) => l.code).filter((code) => !mutedLeagues.includes(code)),
    [leagueOptions, mutedLeagues],
  );

  const visibleLeagues = useMemo(
    () => allLeagues.filter((league) => !mutedLeagues.includes(league.league_code)),
    [allLeagues, mutedLeagues],
  );

  const metrics = useMemo(() => deriveMetrics(visibleLeagues), [visibleLeagues]);
  const signals = useMemo(() => buildSignals(visibleLeagues), [visibleLeagues]);
  const hero = useMemo(() => pickHeroMatch(visibleLeagues), [visibleLeagues]);

  const displayDates = useMemo(() => withToday(availableDates), [availableDates]);

  const handleSelectDate = useCallback((date: string) => {
    setSelectedDate(date);
    setSelectedMatch(null);
    setExpandedId(null);
  }, []);

  const handleToggleLeague = useCallback((code: string) => {
    setMutedLeagues((muted) =>
      muted.includes(code) ? muted.filter((c) => c !== code) : [...muted, code],
    );
  }, []);

  const openDetail = useCallback(
    (match: MatchPrediction, leagueCode: string) => {
      if (isWide) {
        setSelectedMatch(match);
        return;
      }
      const id = buildMatchId(leagueCode, match.home_team, match.away_team);
      navigate(`/match/${id}?date=${encodeURIComponent(selectedDate)}`);
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

  async function handleExport(format: 'csv' | 'excel') {
    setExporting(true);
    try {
      await downloadExport(format, selectedDate);
    } finally {
      setExporting(false);
    }
  }

  async function handleInference() {
    setInferring(true);
    setInferResults(null);
    setInferError(null);
    try {
      setInferResults(await runInference(selectedDate));
    } catch (err) {
      setInferError(err instanceof Error ? err.message : 'Erro na inferência');
    } finally {
      setInferring(false);
    }
  }

  const actions = (
    <>
      <Button variant="outline" size="sm" disabled={exporting} onClick={() => handleExport('csv')}>
        <Download className="w-3.5 h-3.5" />
        CSV
      </Button>
      <Button variant="outline" size="sm" disabled={exporting} onClick={() => handleExport('excel')}>
        <Download className="w-3.5 h-3.5" />
        Excel
      </Button>
      <Button variant="outline" size="sm" disabled={refreshing} onClick={handleRefresh}>
        <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
        Atualizar
      </Button>
      <Button variant="outline" size="sm" disabled={inferring} onClick={handleInference}>
        <Cpu className="w-3.5 h-3.5" />
        Calcular ao vivo
      </Button>
    </>
  );

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <SignalTicker signals={signals} />

      {hero && (
        <MatchOfTheDay
          hero={hero}
          onOpen={() => openDetail(hero.match, hero.leagueCode)}
        />
      )}

      <FilterBar
        dates={displayDates}
        selectedDate={selectedDate}
        onSelectDate={handleSelectDate}
        leagues={leagueOptions}
        activeLeagues={activeLeagues}
        onToggleLeague={handleToggleLeague}
        actions={actions}
      />

      <div className="flex flex-1 overflow-hidden">
        <DashboardRail metrics={metrics} />

        <div className="flex-1 overflow-y-auto min-w-0">
          {loading && (
            <div className="flex items-center justify-center gap-3 py-16 text-fg-muted text-sm">
              <Loader2 className="w-5 h-5 text-accent animate-spin" />
              A carregar previsões…
            </div>
          )}

          {error && !loading && (
            <div className="px-5 md:px-7 py-6">
              <Panel accent="red" className="text-center">
                <p className="text-sm text-accent-red mb-3">{error}</p>
                <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
                  Tentar novamente
                </Button>
              </Panel>
            </div>
          )}

          {(inferError || inferResults) && (
            <div className="px-5 md:px-7 pt-5">
              <Panel accent={inferError ? 'red' : 'accent'} padding="sm">
                <div className="flex items-center gap-2 mb-2">
                  <SectionLabel>Previsões ao vivo (modelo HF)</SectionLabel>
                  <button
                    onClick={() => {
                      setInferResults(null);
                      setInferError(null);
                    }}
                    className="ml-auto text-xs text-fg-subtle hover:text-fg cursor-pointer"
                  >
                    Fechar
                  </button>
                </div>
                {inferError ? (
                  <p className="text-xs text-accent-red">{inferError}</p>
                ) : inferResults && inferResults.length === 0 ? (
                  <p className="text-xs text-fg-subtle">Sem previsões ao vivo para esta data.</p>
                ) : (
                  <div className="flex flex-col">
                    {inferResults?.map((result, i) => (
                      <div
                        key={`${result.home_team}-${result.away_team}-${i}`}
                        className="flex items-center justify-between gap-3 py-1.5 border-b border-line-soft last:border-0"
                      >
                        <span className="text-xs text-fg-muted truncate">
                          {result.home_team} vs {result.away_team}
                        </span>
                        <span className="flex items-center gap-3 shrink-0 font-mono text-[11px]">
                          <span className="text-fg-subtle">{result.league ?? ''}</span>
                          <span className="text-accent">{result.predicted_outcome}</span>
                          <span className="text-fg-subtle">
                            {(result.confidence * 100).toFixed(0)}%
                          </span>
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </Panel>
            </div>
          )}

          {!loading && !error && metrics.matchCount === 0 && (
            <p className="px-5 md:px-7 py-10 text-sm text-fg-subtle">
              Sem jogos para {describeDate(selectedDate)}
              {mutedLeagues.length > 0 ? ' na seleção atual de ligas.' : '.'}
            </p>
          )}

          {!loading &&
            !error &&
            visibleLeagues
              .filter((league) => league.matches.length > 0)
              .map((league) => (
                <section key={league.league_code}>
                  <h2 className="sticky top-0 z-10 bg-bg px-5 md:px-7 pt-4 pb-2 label-mono text-fg-subtle border-b border-line-soft">
                    {league.league_name}
                  </h2>
                  {league.matches.map((match) => {
                    const id = buildMatchId(league.league_code, match.home_team, match.away_team);
                    return (
                      <MatchRow
                        key={id}
                        match={match}
                        expanded={expandedId === id}
                        onToggle={() => setExpandedId(expandedId === id ? null : id)}
                        onOpenDetail={() => openDetail(match, league.league_code)}
                      />
                    );
                  })}
                </section>
              ))}

          <div className="h-10" />
        </div>

        {isWide && <DetailPanel match={selectedMatch} onClose={() => setSelectedMatch(null)} />}
      </div>
    </div>
  );
}
