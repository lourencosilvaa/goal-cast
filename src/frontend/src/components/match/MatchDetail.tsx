import { useState } from 'react';
import { motion } from 'framer-motion';
import { Target, TrendingUp, BarChart3, Sparkles } from 'lucide-react';
import { NeonButton } from '@/components/ui/NeonButton';
import type { MatchPrediction } from '@/types';
import { analyzeMatch } from '@/lib/api';

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formLabel(ppj: number): { text: string; color: string } {
  if (ppj >= 2.0) return { text: 'Boa', color: 'text-accent-green' };
  if (ppj >= 1.5) return { text: 'Média', color: 'text-accent-amber' };
  return { text: 'Fraca', color: 'text-accent-red' };
}

function confidenceClass(conf: string): string {
  if (conf === 'HIGH') return 'bg-accent-green/15 text-accent-green border-accent-green/25';
  if (conf === 'MEDIUM') return 'bg-accent-amber/15 text-accent-amber border-accent-amber/25';
  return 'bg-accent-red/15 text-accent-red border-accent-red/25';
}

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
      const text = await analyzeMatch(match as unknown as Record<string, unknown>, model);
      setAiAnalysis(text);
    } catch (e) {
      setAiError(e instanceof Error ? e.message : 'Erro na análise AI');
    } finally {
      setAiLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          {match.time && (
            <span className="text-xs text-fg-subtle font-mono">{match.time}</span>
          )}
          <span className="text-xs text-fg-subtle">{match.league}</span>
          {hasValueBets && (
            <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-accent-green/15 text-accent-green border border-accent-green/25 font-medium">
              VALUE
            </span>
          )}
        </div>
        <div className="flex items-center justify-between">
          <p className="text-fg font-semibold text-lg flex-1">{match.home_team}</p>
          <span className="px-3 text-fg-subtle text-sm">vs</span>
          <p className="text-fg font-semibold text-lg flex-1 text-right">{match.away_team}</p>
        </div>
      </div>

      {/* Probability bars */}
      <div className="space-y-2">
        <ProbBar label="Casa" value={p.home_win} color="bg-accent-green" />
        <ProbBar label="Empate" value={p.draw} color="bg-accent-amber" />
        <ProbBar label="Fora" value={p.away_win} color="bg-accent-red" />
      </div>

      {/* Prediction */}
      <div className="flex items-center justify-between text-xs text-fg-muted">
        <span>
          Previsão: <span className="text-fg font-medium">{outcomeLabel(match)}</span>
        </span>
        <span>Confiança: {pct(match.confidence)}</span>
      </div>

      {/* Odds */}
      <div className="grid grid-cols-3 gap-2">
        <OddsBadge label="Casa" odds={match.odds?.home ?? null} implied={match.implied_probabilities?.home ?? null} ml={p.home_win} />
        <OddsBadge label="Empate" odds={match.odds?.draw ?? null} implied={match.implied_probabilities?.draw ?? null} ml={p.draw} />
        <OddsBadge label="Fora" odds={match.odds?.away ?? null} implied={match.implied_probabilities?.away ?? null} ml={p.away_win} />
      </div>

      {/* Value bets */}
      {hasValueBets && (
        <div className="space-y-2">
          {match.value_bets.map((vb, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-3 py-2 rounded-xl bg-accent-green/10 border border-accent-green/20"
            >
              <div className="flex items-center gap-2">
                <TrendingUp className="w-3.5 h-3.5 text-accent-green" />
                <span className="text-sm text-fg">{vb.outcome}</span>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-accent-green">Edge {vb.edge_pct}</span>
                <span className="text-fg-muted">Kelly {pct(vb.kelly_fraction)}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${confidenceClass(vb.confidence)}`}>
                  {vb.confidence}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Expected Goals */}
      {match.expected_goals && (
        <Section icon={<Target className="w-3.5 h-3.5" />} title="Golos Esperados (xG)">
          <div className="grid grid-cols-3 gap-2">
            <StatBox label={match.home_team} value={match.expected_goals.home.toFixed(1)} />
            <StatBox label="Total" value={match.expected_goals.total.toFixed(1)} accent="blue" />
            <StatBox label={match.away_team} value={match.expected_goals.away.toFixed(1)} />
          </div>
        </Section>
      )}

      {/* Over/Under */}
      {match.over_under && (
        <Section icon={<BarChart3 className="w-3.5 h-3.5" />} title="Mercado de Golos">
          <div className="grid grid-cols-4 gap-2">
            <StatBox label="Over 1.5" value={pct(match.over_under.over_15)} accent="blue" />
            <StatBox label="Over 2.5" value={pct(match.over_under.over_25)} accent="blue" />
            <StatBox label="Over 3.5" value={pct(match.over_under.over_35)} accent="blue" />
            <StatBox label="Under 2.5" value={pct(match.over_under.under_25)} accent="blue" />
          </div>
        </Section>
      )}

      {/* BTTS */}
      {match.btts && (
        <Section title="Ambas Marcam">
          <div className="grid grid-cols-2 gap-2">
            <StatBox label="Sim" value={pct(match.btts.yes)} accent={match.btts.yes > 0.5 ? 'green' : undefined} />
            <StatBox label="Não" value={pct(match.btts.no)} accent={match.btts.no > 0.5 ? 'green' : undefined} />
          </div>
        </Section>
      )}

      {/* Scorelines */}
      {match.top_scorelines && match.top_scorelines.length > 0 && (
        <Section title="Resultados Mais Prováveis">
          <div className="flex gap-2 flex-wrap">
            {match.top_scorelines.map((s, i) => (
              <span
                key={i}
                className={`text-xs px-2.5 py-1 rounded-lg border ${
                  i === 0
                    ? 'bg-accent-green/15 text-accent-green border-accent-green/25'
                    : 'bg-card-2 text-fg-muted border-line'
                }`}
              >
                {s.score} ({pct(s.prob)})
              </span>
            ))}
          </div>
        </Section>
      )}

      {/* Form */}
      {match.form && (
        <Section title="Forma Recente (ppj)">
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-card-2 border border-line">
              <span className="text-xs text-fg-muted">{match.home_team}</span>
              <span className={`text-sm font-medium ${formLabel(match.form.home).color}`}>
                {match.form.home.toFixed(1)} — {formLabel(match.form.home).text}
              </span>
            </div>
            <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-card-2 border border-line">
              <span className="text-xs text-fg-muted">{match.away_team}</span>
              <span className={`text-sm font-medium ${formLabel(match.form.away).color}`}>
                {match.form.away.toFixed(1)} — {formLabel(match.form.away).text}
              </span>
            </div>
          </div>
        </Section>
      )}

      {/* AI Analysis */}
      <div className="pt-2 border-t border-line">
        {!aiAnalysis ? (
          <NeonButton
            variant="secondary"
            size="sm"
            loading={aiLoading}
            onClick={handleAiAnalysis}
            className="w-full"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5 inline text-accent-purple" />
            Análise AI (Gemini)
          </NeonButton>
        ) : (
          <div className="text-sm text-fg-muted leading-relaxed whitespace-pre-line">
            {aiAnalysis}
          </div>
        )}
        {aiError && <p className="text-xs text-accent-red mt-2">{aiError}</p>}
      </div>
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon?: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h4 className="text-xs text-fg-subtle uppercase tracking-wider mb-2 flex items-center gap-1.5">
        {icon}
        {title}
      </h4>
      {children}
    </div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-fg-muted w-12">{label}</span>
      <div className="flex-1 h-2 bg-card-2 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${(value * 100).toFixed(1)}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>
      <span className="text-xs text-fg-muted w-12 text-right font-mono">{pct(value)}</span>
    </div>
  );
}

function OddsBadge({
  label,
  odds,
  implied,
  ml,
}: {
  label: string;
  odds: number | null;
  implied: number | null;
  ml: number;
}) {
  const hasOdds = odds != null && implied != null;
  const isValue = hasOdds && ml > implied + 0.03;
  return (
    <div
      className={`text-center px-2 py-2 rounded-xl border ${
        isValue ? 'bg-accent-green/10 border-accent-green/25' : 'bg-card-2 border-line'
      }`}
    >
      <p className="text-[10px] text-fg-subtle mb-0.5">{label}</p>
      {hasOdds ? (
        <>
          <p className="text-sm font-semibold text-fg">{odds.toFixed(2)}</p>
          <p className="text-[10px] text-fg-subtle">B365 {pct(implied)}</p>
          {isValue && <p className="text-[10px] text-accent-green font-medium mt-0.5">ML {pct(ml)}</p>}
        </>
      ) : (
        <p className="text-sm font-semibold text-fg-subtle">N/A</p>
      )}
    </div>
  );
}

function StatBox({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: 'green' | 'blue';
}) {
  const accentClass =
    accent === 'green'
      ? 'bg-accent-green/10 border-accent-green/25'
      : accent === 'blue'
        ? 'bg-accent-blue/10 border-accent-blue/25'
        : 'bg-card-2 border-line';
  return (
    <div className={`text-center px-2 py-2 rounded-xl border ${accentClass}`}>
      <p className="text-[10px] text-fg-subtle mb-0.5">{label}</p>
      <p className="text-sm font-semibold text-fg">{value}</p>
    </div>
  );
}
