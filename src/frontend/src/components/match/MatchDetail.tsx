import { useState } from 'react';
import clsx from 'clsx';
import { Button } from '@/components/ui/Button';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { ProbabilityBar, ProbabilityLegend } from '@/components/ui/ProbabilityBar';
import type { MatchPrediction } from '@/types';
import { analyzeMatch } from '@/lib/api';

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formLabel(pointsPerGame: number): { text: string; color: string } {
  if (pointsPerGame >= 2.0) return { text: 'Boa', color: 'text-accent-green' };
  if (pointsPerGame >= 1.5) return { text: 'Média', color: 'text-accent-amber' };
  return { text: 'Fraca', color: 'text-accent-red' };
}

const CONFIDENCE_CLASS: Record<string, string> = {
  HIGH: 'text-accent-green border-accent-green/40',
  MEDIUM: 'text-accent-amber border-accent-amber/40',
  LOW: 'text-fg-subtle border-line',
};

function outcomeLabel(match: MatchPrediction): string {
  if (match.predicted_outcome === 'Home Win') return `Vitória ${match.home_team}`;
  if (match.predicted_outcome === 'Away Win') return `Vitória ${match.away_team}`;
  return 'Empate';
}

interface MatchDetailProps {
  match: MatchPrediction;
}

/** Full detail view for a single match. Shared by the docked panel and sub-page. */
export function MatchDetail({ match }: MatchDetailProps) {
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  const p = match.probabilities;
  const hasValueBets = match.value_bets.length > 0;

  async function handleAiAnalysis() {
    const model = localStorage.getItem('gemini_model') || undefined;
    setAiLoading(true);
    setAiError(null);
    try {
      setAiAnalysis(await analyzeMatch(match as unknown as Record<string, unknown>, model));
    } catch (e) {
      setAiError(e instanceof Error ? e.message : 'Erro na análise AI');
    } finally {
      setAiLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="flex items-center gap-2 mb-2.5">
          {match.time && <span className="font-mono text-[11px] text-fg-subtle">{match.time}</span>}
          <SectionLabel>{match.league}</SectionLabel>
          {hasValueBets && (
            <span className="ml-auto px-1.5 py-0.5 rounded border border-accent-green/40 font-mono text-[9px] font-bold text-accent-green">
              VALUE
            </span>
          )}
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="flex-1 text-base font-bold text-fg">{match.home_team}</span>
          <span className="text-xs text-fg-subtle shrink-0">vs</span>
          <span className="flex-1 text-base font-bold text-fg text-right">{match.away_team}</span>
        </div>
      </div>

      <div>
        <ProbabilityLegend
          probabilities={p}
          homeLabel={match.home_team}
          awayLabel={match.away_team}
          className="mb-1.5"
        />
        <ProbabilityBar probabilities={p} />
      </div>

      <div className="flex items-center justify-between font-mono text-xs">
        <span className="text-fg-muted">{outcomeLabel(match)}</span>
        <span className="text-fg-subtle">Confiança {pct(match.confidence)}</span>
      </div>

      <Section title="Odds">
        <div className="grid grid-cols-3 gap-2">
          <OddsBox
            label="Casa"
            odds={match.odds?.home ?? null}
            implied={match.implied_probabilities?.home ?? null}
            model={p.home_win}
          />
          <OddsBox
            label="Empate"
            odds={match.odds?.draw ?? null}
            implied={match.implied_probabilities?.draw ?? null}
            model={p.draw}
          />
          <OddsBox
            label="Fora"
            odds={match.odds?.away ?? null}
            implied={match.implied_probabilities?.away ?? null}
            model={p.away_win}
          />
        </div>
      </Section>

      {hasValueBets && (
        <Section title="Apostas de valor">
          <div className="flex flex-col gap-2">
            {match.value_bets.map((bet, i) => (
              <div
                key={i}
                className="flex items-center justify-between gap-2 px-3 py-2 rounded-md border border-accent-green/40"
              >
                <span className="text-[13px] text-fg truncate">{bet.outcome}</span>
                <span className="flex items-center gap-2.5 shrink-0 font-mono text-[11px]">
                  <span className="text-accent-green">Edge {bet.edge_pct}</span>
                  <span className="text-fg-subtle">Kelly {pct(bet.kelly_fraction)}</span>
                  <span
                    className={clsx(
                      'px-1.5 py-0.5 rounded border font-bold text-[9px]',
                      CONFIDENCE_CLASS[bet.confidence] ?? CONFIDENCE_CLASS.LOW,
                    )}
                  >
                    {bet.confidence}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {match.expected_goals && (
        <Section title="Golos esperados (xG)">
          <div className="grid grid-cols-3 gap-2">
            <StatBox label={match.home_team} value={match.expected_goals.home.toFixed(1)} />
            <StatBox label="Total" value={match.expected_goals.total.toFixed(1)} highlight />
            <StatBox label={match.away_team} value={match.expected_goals.away.toFixed(1)} />
          </div>
        </Section>
      )}

      {match.over_under && (
        <Section title="Mercado de golos">
          <div className="grid grid-cols-4 gap-2">
            <StatBox label="Over 1.5" value={pct(match.over_under.over_15)} />
            <StatBox label="Over 2.5" value={pct(match.over_under.over_25)} />
            <StatBox label="Over 3.5" value={pct(match.over_under.over_35)} />
            <StatBox label="Under 2.5" value={pct(match.over_under.under_25)} />
          </div>
        </Section>
      )}

      {match.btts && (
        <Section title="Ambas marcam">
          <div className="grid grid-cols-2 gap-2">
            <StatBox label="Sim" value={pct(match.btts.yes)} highlight={match.btts.yes > 0.5} />
            <StatBox label="Não" value={pct(match.btts.no)} highlight={match.btts.no > 0.5} />
          </div>
        </Section>
      )}

      {match.top_scorelines && match.top_scorelines.length > 0 && (
        <Section title="Resultados mais prováveis">
          <div className="flex gap-1.5 flex-wrap">
            {match.top_scorelines.map((line, i) => (
              <span
                key={line.score}
                className={clsx(
                  'px-2 py-0.5 rounded border font-mono text-[11px]',
                  i === 0 ? 'border-accent/45 text-accent' : 'border-line text-fg-muted',
                )}
              >
                {line.score} · {pct(line.prob)}
              </span>
            ))}
          </div>
        </Section>
      )}

      {match.form && (
        <Section title="Forma recente (ppj)">
          <div className="grid grid-cols-2 gap-2">
            {(
              [
                [match.home_team, match.form.home],
                [match.away_team, match.form.away],
              ] as const
            ).map(([team, value]) => (
              <div
                key={team}
                className="flex items-center justify-between gap-2 px-3 py-2 rounded-md border border-line"
              >
                <span className="text-[11px] text-fg-subtle truncate">{team}</span>
                <span className={`font-mono text-xs font-semibold shrink-0 ${formLabel(value).color}`}>
                  {value.toFixed(1)} {formLabel(value).text}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <div className="pt-1 border-t border-line-soft">
        {!aiAnalysis ? (
          <Button variant="outline" size="sm" loading={aiLoading} onClick={handleAiAnalysis} className="w-full">
            Análise AI (Gemini)
          </Button>
        ) : (
          <p className="text-[13px] text-fg-muted leading-relaxed whitespace-pre-line">
            {aiAnalysis}
          </p>
        )}
        {aiError && <p className="text-xs text-accent-red mt-2">{aiError}</p>}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <SectionLabel className="mb-2">{title}</SectionLabel>
      {children}
    </div>
  );
}

function OddsBox({
  label,
  odds,
  implied,
  model,
}: {
  label: string;
  odds: number | null;
  implied: number | null;
  model: number;
}) {
  const hasOdds = odds != null && implied != null;
  // A three-point gap is the same threshold the backend uses to flag value —
  // matching it keeps the panel from disagreeing with the VALUE tag beside it.
  const isValue = hasOdds && model > implied + 0.03;

  return (
    <div
      className={clsx(
        'text-center px-2 py-2 rounded-md border',
        isValue ? 'border-accent-green/40' : 'border-line',
      )}
    >
      <p className="text-[10px] text-fg-subtle mb-0.5 truncate">{label}</p>
      {hasOdds ? (
        <>
          <p className="font-mono text-[13px] font-semibold text-fg">{odds.toFixed(2)}</p>
          <p className="font-mono text-[10px] text-fg-subtle">B365 {pct(implied)}</p>
          {isValue && (
            <p className="font-mono text-[10px] font-semibold text-accent-green mt-0.5">
              Modelo {pct(model)}
            </p>
          )}
        </>
      ) : (
        <p className="font-mono text-[13px] text-fg-subtle">N/D</p>
      )}
    </div>
  );
}

function StatBox({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={clsx(
        'text-center px-2 py-2 rounded-md border',
        highlight ? 'border-accent/45' : 'border-line',
      )}
    >
      <p className="text-[10px] text-fg-subtle mb-0.5 truncate">{label}</p>
      <p className={clsx('font-mono text-[13px] font-semibold', highlight ? 'text-accent' : 'text-fg')}>
        {value}
      </p>
    </div>
  );
}
