import { useMemo } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { PageBody } from '@/components/layout/PageBody';
import { MatchDetail } from '@/components/match/MatchDetail';
import { usePredictions } from '@/contexts/PredictionsContext';
import { findMatchById } from '@/lib/matchId';
import { todayStr } from '@/lib/dates';

/** Standalone match detail page — used on narrow screens and for deep links. */
export function MatchDetailPage() {
  const { matchId = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const date = searchParams.get('date') ?? todayStr();

  const { data: leagues, loading } = usePredictions(date);

  const resolved = useMemo(() => findMatchById(leagues, matchId), [leagues, matchId]);

  return (
    <PageBody maxWidth="md">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1.5 text-xs font-semibold text-fg-subtle hover:text-fg mb-4 transition-colors cursor-pointer"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        Voltar
      </button>

      {loading && !resolved ? (
        <div className="flex items-center justify-center gap-3 py-16 text-fg-muted text-sm">
          <Loader2 className="w-5 h-5 text-accent animate-spin" />
          A carregar…
        </div>
      ) : resolved ? (
        <Panel accent={resolved.match.value_bets.length > 0 ? 'green' : 'neutral'}>
          <SectionLabel className="mb-4">{resolved.leagueName}</SectionLabel>
          <MatchDetail match={resolved.match} />
        </Panel>
      ) : (
        <Panel className="text-center">
          <p className="text-sm text-fg-muted mb-4">Jogo não encontrado.</p>
          <Button variant="outline" size="sm" onClick={() => navigate('/')}>
            Ir para o Dashboard
          </Button>
        </Panel>
      )}
    </PageBody>
  );
}
