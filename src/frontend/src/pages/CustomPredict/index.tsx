import { useState } from 'react';
import { motion } from 'framer-motion';
import { Wand2, Loader2 } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { predictCustom, type CustomPrediction } from '@/lib/api';

const LEAGUES = [
  { code: 'E0', name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
  { code: 'SP1', name: 'La Liga', flag: '🇪🇸' },
  { code: 'D1', name: 'Bundesliga', flag: '🇩🇪' },
  { code: 'I1', name: 'Serie A', flag: '🇮🇹' },
  { code: 'F1', name: 'Ligue 1', flag: '🇫🇷' },
  { code: 'P1', name: 'Liga Portugal', flag: '🇵🇹' },
];

const OUTCOME_LABELS: Record<string, string> = {
  'Home Win': 'Vitória Casa',
  'Draw': 'Empate',
  'Away Win': 'Vitória Fora',
};

const OUTCOME_COLORS: Record<string, string> = {
  'Home Win': 'text-green-400',
  'Draw': 'text-yellow-400',
  'Away Win': 'text-blue-400',
};

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-white/50">{label}</span>
        <span className={`font-semibold ${color}`}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          className={`h-full rounded-full bg-current ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

export function CustomPredictPage() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [leagueCode, setLeagueCode] = useState('E0');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CustomPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    if (!homeTeam.trim() || !awayTeam.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const prediction = await predictCustom(homeTeam.trim(), awayTeam.trim(), leagueCode);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao obter previsão');
    } finally {
      setLoading(false);
    }
  }

  const inputClass =
    'w-full px-4 py-3 rounded-xl bg-white/[0.03] border border-white/[0.08] text-white/90 placeholder-white/20 focus:outline-none focus:border-green-500/40 transition-colors text-sm';

  return (
    <div className="max-w-lg mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white/90">Previsão Personalizada</h1>
        <p className="text-sm text-white/40 mt-1">
          Escolhe duas equipas e obtém uma previsão do modelo ML
        </p>
      </div>

      <GlassCard className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs text-white/40 font-medium uppercase tracking-wide">
              Equipa Casa
            </label>
            <input
              value={homeTeam}
              onChange={(e) => setHomeTeam(e.target.value)}
              placeholder="ex: Sporting CP"
              className={inputClass}
              onKeyDown={(e) => e.key === 'Enter' && handlePredict()}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-white/40 font-medium uppercase tracking-wide">
              Equipa Fora
            </label>
            <input
              value={awayTeam}
              onChange={(e) => setAwayTeam(e.target.value)}
              placeholder="ex: Tondela"
              className={inputClass}
              onKeyDown={(e) => e.key === 'Enter' && handlePredict()}
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs text-white/40 font-medium uppercase tracking-wide">
            Liga
          </label>
          <select
            value={leagueCode}
            onChange={(e) => setLeagueCode(e.target.value)}
            className={`${inputClass} cursor-pointer`}
          >
            {LEAGUES.map((l) => (
              <option key={l.code} value={l.code} className="bg-[#0e0e16]">
                {l.flag} {l.name}
              </option>
            ))}
          </select>
        </div>

        <NeonButton
          variant="primary"
          size="md"
          loading={loading}
          onClick={handlePredict}
          disabled={!homeTeam.trim() || !awayTeam.trim()}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
              A calcular...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4 mr-2 inline" />
              Prever
            </>
          )}
        </NeonButton>
      </GlassCard>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4"
        >
          <GlassCard className="text-center py-4">
            <p className="text-red-400 text-sm">✗ {error}</p>
          </GlassCard>
        </motion.div>
      )}

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6"
        >
          <GlassCard className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-white/30 mb-1">{result.league}</p>
                <p className="text-lg font-bold text-white/90">
                  {result.home_team}{' '}
                  <span className="text-white/30 font-normal">vs</span>{' '}
                  {result.away_team}
                </p>
              </div>
              <div className="text-right">
                <p className={`text-xl font-bold ${OUTCOME_COLORS[result.predicted_outcome] ?? 'text-white'}`}>
                  {OUTCOME_LABELS[result.predicted_outcome] ?? result.predicted_outcome}
                </p>
                <p className="text-xs text-white/30 mt-0.5">
                  {(result.confidence * 100).toFixed(0)}% confiança
                </p>
              </div>
            </div>

            <div className="space-y-3 pt-2 border-t border-white/[0.06]">
              <ProbBar
                label="Vitória Casa"
                value={result.probabilities.home_win}
                color="text-green-400"
              />
              <ProbBar
                label="Empate"
                value={result.probabilities.draw}
                color="text-yellow-400"
              />
              <ProbBar
                label="Vitória Fora"
                value={result.probabilities.away_win}
                color="text-blue-400"
              />
            </div>

            <p className="text-[10px] text-white/20 text-center">
              Previsão baseada em dados históricos · Aposte com responsabilidade
            </p>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
