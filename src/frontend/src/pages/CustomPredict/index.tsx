import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Wand2, Loader2, ChevronDown } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { predictCustom, fetchTeams, type CustomPrediction } from '@/lib/api';

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
  'Home Win': 'text-accent-green',
  'Draw': 'text-accent-amber',
  'Away Win': 'text-accent-blue',
};

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-fg-muted">{label}</span>
        <span className={`font-semibold ${color}`}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-card-2 overflow-hidden">
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

function TeamCombobox({
  label,
  value,
  onChange,
  teams,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  teams: string[];
  placeholder: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  const filtered = teams.filter((t) =>
    t.toLowerCase().includes(query.toLowerCase()),
  );

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  const inputClass =
    'w-full px-4 py-3 rounded-xl bg-card-2 border border-line text-fg placeholder-fg-subtle focus:outline-none focus:border-accent-blue/40 transition-colors text-sm';

  return (
    <div className="space-y-1.5" ref={ref}>
      <label className="text-xs text-fg-muted font-medium uppercase tracking-wide">
        {label}
      </label>
      <div className="relative">
        <input
          value={open ? query : value}
          onChange={(e) => {
            setQuery(e.target.value);
            if (!open) setOpen(true);
          }}
          onFocus={() => {
            setOpen(true);
            setQuery('');
          }}
          placeholder={placeholder}
          className={inputClass}
        />
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-subtle pointer-events-none" />
        {open && filtered.length > 0 && (
          <div className="absolute z-50 mt-1 w-full max-h-48 overflow-y-auto rounded-xl bg-card border border-line shadow-xl">
            {filtered.map((team) => (
              <button
                key={team}
                type="button"
                className={`w-full text-left px-4 py-2 text-sm hover:bg-card-2 transition-colors ${
                  team === value ? 'text-accent-green' : 'text-fg'
                }`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(team);
                  setQuery('');
                  setOpen(false);
                }}
              >
                {team}
              </button>
            ))}
          </div>
        )}
        {open && filtered.length === 0 && query && (
          <div className="absolute z-50 mt-1 w-full rounded-xl bg-card border border-line shadow-xl px-4 py-3">
            <p className="text-xs text-fg-subtle">Nenhuma equipa encontrada</p>
          </div>
        )}
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
  const [teamsByLeague, setTeamsByLeague] = useState<Record<string, string[]>>({});

  useEffect(() => {
    fetchTeams()
      .then(setTeamsByLeague)
      .catch(() => {});
  }, []);

  const currentTeams = teamsByLeague[leagueCode] ?? [];

  function handleLeagueChange(code: string) {
    setLeagueCode(code);
    setHomeTeam('');
    setAwayTeam('');
    setResult(null);
    setError(null);
  }

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
    'w-full px-4 py-3 rounded-xl bg-card-2 border border-line text-fg placeholder-fg-subtle focus:outline-none focus:border-accent-blue/40 transition-colors text-sm';

  return (
    <div className="max-w-lg mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-fg">Previsão Personalizada</h1>
        <p className="text-sm text-fg-muted mt-1">
          Escolhe duas equipas e obtém uma previsão do modelo ML
        </p>
      </div>

      <GlassCard className="space-y-5">
        <div className="space-y-1.5">
          <label className="text-xs text-fg-muted font-medium uppercase tracking-wide">
            Liga
          </label>
          <select
            value={leagueCode}
            onChange={(e) => handleLeagueChange(e.target.value)}
            className={`${inputClass} cursor-pointer`}
          >
            {LEAGUES.map((l) => (
              <option key={l.code} value={l.code} className="bg-card text-fg">
                {l.flag} {l.name}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <TeamCombobox
            label="Equipa Casa"
            value={homeTeam}
            onChange={setHomeTeam}
            teams={currentTeams.filter((t) => t !== awayTeam)}
            placeholder="Selecionar..."
          />
          <TeamCombobox
            label="Equipa Fora"
            value={awayTeam}
            onChange={setAwayTeam}
            teams={currentTeams.filter((t) => t !== homeTeam)}
            placeholder="Selecionar..."
          />
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
            <p className="text-accent-red text-sm">✗ {error}</p>
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
                <p className="text-xs text-fg-subtle mb-1">{result.league}</p>
                <p className="text-lg font-bold text-fg">
                  {result.home_team}{' '}
                  <span className="text-fg-subtle font-normal">vs</span>{' '}
                  {result.away_team}
                </p>
              </div>
              <div className="text-right">
                <p className={`text-xl font-bold ${OUTCOME_COLORS[result.predicted_outcome] ?? 'text-fg'}`}>
                  {OUTCOME_LABELS[result.predicted_outcome] ?? result.predicted_outcome}
                </p>
                <p className="text-xs text-fg-subtle mt-0.5">
                  {(result.confidence * 100).toFixed(0)}% confiança
                </p>
              </div>
            </div>

            <div className="space-y-3 pt-2 border-t border-line">
              <ProbBar
                label="Vitória Casa"
                value={result.probabilities.home_win}
                color="text-accent-green"
              />
              <ProbBar
                label="Empate"
                value={result.probabilities.draw}
                color="text-accent-amber"
              />
              <ProbBar
                label="Vitória Fora"
                value={result.probabilities.away_win}
                color="text-accent-blue"
              />
            </div>

            <p className="text-[10px] text-fg-subtle text-center">
              Previsão baseada em dados históricos · Aposte com responsabilidade
            </p>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
