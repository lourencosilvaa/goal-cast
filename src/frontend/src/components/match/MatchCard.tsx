import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Target,
  TrendingUp,
  BarChart3,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import type { MatchPrediction } from '@/types';
import { analyzeMatch } from '@/lib/api';

interface MatchCardProps {
  match: MatchPrediction;
  index: number;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formLabel(ppj: number): { text: string; color: string } {
  if (ppj >= 2.0) return { text: 'Boa', color: 'text-emerald-400' };
  if (ppj >= 1.5) return { text: 'Média', color: 'text-amber-400' };
  return { text: 'Fraca', color: 'text-red-400' };
}

function confidenceColor(conf: string): string {
  if (conf === 'HIGH') return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/20';
  if (conf === 'MEDIUM') return 'bg-amber-500/15 text-amber-300 border-amber-500/20';
  return 'bg-red-500/15 text-red-300 border-red-500/20';
}

export function MatchCard({ match, index }: MatchCardProps) {
  const [expanded, setExpanded] = useState(false);
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
    >
      <GlassCard
        gradient={hasValueBets ? 'green' : undefined}
        className={hasValueBets ? 'neon-glow' : ''}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            {match.time && (
              <span className="text-xs text-white/40 font-mono">{match.time}</span>
            )}
            {hasValueBets && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/15 text-green-300 border border-green-500/20 font-medium">
                VALUE
              </span>
            )}
          </div>
          <span className="text-xs text-white/30">{match.league}</span>
        </div>

        {/* Teams */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex-1">
            <p className="text-white/90 font-semibold text-lg">{match.home_team}</p>
          </div>
          <div className="px-4 text-white/20 text-sm font-medium">vs</div>
          <div className="flex-1 text-right">
            <p className="text-white/90 font-semibold text-lg">{match.away_team}</p>
          </div>
        </div>

        {/* Probability bars */}
        <div className="space-y-2 mb-4">
          <ProbBar label="Casa" value={p.home_win} color="bg-emerald-500" />
          <ProbBar label="Empate" value={p.draw} color="bg-amber-500" />
          <ProbBar label="Fora" value={p.away_win} color="bg-red-500" />
        </div>

        {/* Prediction */}
        <div className="flex items-center justify-between text-xs text-white/50 mb-3">
          <span>
            Previsão:{' '}
            <span className="text-white/80 font-medium">
              {match.predicted_outcome === 'Home Win'
                ? `Vitória ${match.home_team}`
                : match.predicted_outcome === 'Away Win'
                  ? `Vitória ${match.away_team}`
                  : 'Empate'}
            </span>
          </span>
          <span>Confiança: {pct(match.confidence)}</span>
        </div>

        {/* Odds */}
        <div className="grid grid-cols-3 gap-2 mb-4">
          <OddsBadge label="Casa" odds={match.odds?.home ?? null} implied={match.implied_probabilities?.home ?? null} ml={p.home_win} />
          <OddsBadge label="Empate" odds={match.odds?.draw ?? null} implied={match.implied_probabilities?.draw ?? null} ml={p.draw} />
          <OddsBadge label="Fora" odds={match.odds?.away ?? null} implied={match.implied_probabilities?.away ?? null} ml={p.away_win} />
        </div>

        {/* Value bets */}
        {hasValueBets && (
          <div className="space-y-2 mb-4">
            {match.value_bets.map((vb, i) => (
              <div
                key={i}
                className="flex items-center justify-between px-3 py-2 rounded-xl bg-green-500/10 border border-green-500/15"
              >
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-3.5 h-3.5 text-green-400" />
                  <span className="text-sm text-white/80">{vb.outcome}</span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-green-300">Edge {vb.edge_pct}</span>
                  <span className="text-white/40">Kelly {pct(vb.kelly_fraction)}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${confidenceColor(vb.confidence)}`}
                  >
                    {vb.confidence}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-center gap-1 text-xs text-white/30 hover:text-white/60 transition-colors py-1 cursor-pointer"
        >
          {expanded ? (
            <>Menos detalhes <ChevronUp className="w-3.5 h-3.5" /></>
          ) : (
            <>Mais detalhes <ChevronDown className="w-3.5 h-3.5" /></>
          )}
        </button>

        {/* Expanded details */}
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 space-y-4"
          >
            {/* Expected Goals */}
            {match.expected_goals && (
              <div>
                <h4 className="text-xs text-white/40 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Target className="w-3.5 h-3.5" /> Golos Esperados (xG)
                </h4>
                <div className="grid grid-cols-3 gap-2">
                  <StatBox label={match.home_team} value={match.expected_goals.home.toFixed(1)} />
                  <StatBox label="Total" value={match.expected_goals.total.toFixed(1)} />
                  <StatBox label={match.away_team} value={match.expected_goals.away.toFixed(1)} />
                </div>
              </div>
            )}

            {/* Over/Under */}
            {match.over_under && (
              <div>
                <h4 className="text-xs text-white/40 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <BarChart3 className="w-3.5 h-3.5" /> Mercado de Golos
                </h4>
                <div className="grid grid-cols-4 gap-2">
                  <StatBox label="Over 1.5" value={pct(match.over_under.over_15)} />
                  <StatBox label="Over 2.5" value={pct(match.over_under.over_25)} />
                  <StatBox label="Over 3.5" value={pct(match.over_under.over_35)} />
                  <StatBox label="Under 2.5" value={pct(match.over_under.under_25)} />
                </div>
              </div>
            )}

            {/* BTTS */}
            {match.btts && (
              <div>
                <h4 className="text-xs text-white/40 uppercase tracking-wider mb-2">
                  Ambas Marcam
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  <StatBox label="Sim" value={pct(match.btts.yes)} highlight={match.btts.yes > 0.5} />
                  <StatBox label="Não" value={pct(match.btts.no)} highlight={match.btts.no > 0.5} />
                </div>
              </div>
            )}

            {/* Scorelines */}
            {match.top_scorelines && match.top_scorelines.length > 0 && (
              <div>
                <h4 className="text-xs text-white/40 uppercase tracking-wider mb-2">
                  Resultados Mais Prováveis
                </h4>
                <div className="flex gap-2 flex-wrap">
                  {match.top_scorelines.map((s, i) => (
                    <span
                      key={i}
                      className={`text-xs px-2.5 py-1 rounded-lg border ${
                        i === 0
                          ? 'bg-green-500/15 text-green-300 border-green-500/20'
                          : 'bg-white/5 text-white/60 border-white/[0.06]'
                      }`}
                    >
                      {s.score} ({pct(s.prob)})
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Form */}
            {match.form && (
              <div>
                <h4 className="text-xs text-white/40 uppercase tracking-wider mb-2">
                  Forma Recente (ppj)
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-white/5 border border-white/[0.06]">
                    <span className="text-xs text-white/50">{match.home_team}</span>
                    <span className={`text-sm font-medium ${formLabel(match.form.home).color}`}>
                      {match.form.home.toFixed(1)} — {formLabel(match.form.home).text}
                    </span>
                  </div>
                  <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-white/5 border border-white/[0.06]">
                    <span className="text-xs text-white/50">{match.away_team}</span>
                    <span className={`text-sm font-medium ${formLabel(match.form.away).color}`}>
                      {match.form.away.toFixed(1)} — {formLabel(match.form.away).text}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* AI Analysis */}
            <div className="pt-2 border-t border-white/[0.06]">
              {!aiAnalysis ? (
                <NeonButton
                  variant="secondary"
                  size="sm"
                  loading={aiLoading}
                  onClick={handleAiAnalysis}
                  className="w-full"
                >
                  <Sparkles className="w-3.5 h-3.5 mr-1.5 inline" />
                  Análise AI (Gemini)
                </NeonButton>
              ) : (
                <div className="text-sm text-white/70 leading-relaxed whitespace-pre-line">
                  {aiAnalysis}
                </div>
              )}
              {aiError && (
                <p className="text-xs text-red-400 mt-2">{aiError}</p>
              )}
            </div>
          </motion.div>
        )}
      </GlassCard>
    </motion.div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-white/40 w-12">{label}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bar-animate ${color}`}
          style={{ width: `${(value * 100).toFixed(1)}%` }}
        />
      </div>
      <span className="text-xs text-white/60 w-12 text-right font-mono">
        {pct(value)}
      </span>
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
        isValue
          ? 'bg-green-500/10 border-green-500/20'
          : 'bg-white/5 border-white/[0.06]'
      }`}
    >
      <p className="text-[10px] text-white/30 mb-0.5">{label}</p>
      {hasOdds ? (
        <>
          <p className="text-sm font-semibold text-white/80">
            {odds.toFixed(2)}
          </p>
          <p className="text-[10px] text-white/30">B365 {pct(implied)}</p>
          {isValue && (
            <p className="text-[10px] text-green-400 font-medium mt-0.5">
              ML {pct(ml)}
            </p>
          )}
        </>
      ) : (
        <p className="text-sm font-semibold text-white/40">N/A</p>
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
      className={`text-center px-2 py-2 rounded-xl border ${
        highlight
          ? 'bg-emerald-500/10 border-emerald-500/20'
          : 'bg-white/5 border-white/[0.06]'
      }`}
    >
      <p className="text-[10px] text-white/30 mb-0.5">{label}</p>
      <p className="text-sm font-semibold text-white/80">{value}</p>
    </div>
  );
}
