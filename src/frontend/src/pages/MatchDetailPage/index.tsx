import { useMemo } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Loader2, CircleSlash } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { MatchDetail } from '@/components/match/MatchDetail';
import { usePredictions } from '@/contexts/PredictionsContext';
import { findMatchById } from '@/lib/matchId';

function todayStr(): string {
  const d = new Date();
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  return `${dd}/${mm}/${d.getFullYear()}`;
}

/** Standalone match detail page — used on narrow screens and for deep links. */
export function MatchDetailPage() {
  const { matchId = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const date = searchParams.get('date') ?? todayStr();

  const { data: leagues, loading } = usePredictions(date);

  const resolved = useMemo(() => findMatchById(leagues, matchId), [leagues, matchId]);

  return (
    <div className="max-w-2xl mx-auto">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1.5 text-sm text-fg-muted hover:text-fg mb-4 transition-colors cursor-pointer"
      >
        <ArrowLeft className="w-4 h-4" />
        Voltar
      </button>

      {loading && !resolved ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-accent-blue animate-spin" />
        </div>
      ) : resolved ? (
        <GlassCard accent={resolved.match.value_bets.length > 0 ? 'green' : 'neutral'}>
          <p className="text-xs text-fg-subtle mb-4">{resolved.leagueName}</p>
          <MatchDetail match={resolved.match} />
        </GlassCard>
      ) : (
        <GlassCard className="text-center py-10">
          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-card-2 flex items-center justify-center">
            <CircleSlash className="w-6 h-6 text-fg-subtle" />
          </div>
          <p className="text-fg-muted mb-4">Jogo não encontrado.</p>
          <NeonButton variant="secondary" size="sm" onClick={() => navigate('/')}>
            Ir para o Dashboard
          </NeonButton>
        </GlassCard>
      )}
    </div>
  );
}
