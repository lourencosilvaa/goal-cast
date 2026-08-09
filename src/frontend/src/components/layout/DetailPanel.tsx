import { X } from 'lucide-react';
import type { MatchPrediction } from '@/types';
import { MatchDetail } from '@/components/match/MatchDetail';

interface DetailPanelProps {
  match: MatchPrediction | null;
  onClose: () => void;
}

/** Docked right-hand panel that shows the full match detail on wide screens. */
export function DetailPanel({ match, onClose }: DetailPanelProps) {
  if (!match) return null;

  return (
    <aside className="w-[400px] shrink-0 h-full border-l border-line overflow-y-auto scrollbar-hide bg-card">
      <div className="flex items-center justify-between px-5 py-3 sticky top-0 z-10 bg-card border-b border-line-soft">
        <span className="label-mono text-fg-subtle">Detalhe do jogo</span>
        <button
          onClick={onClose}
          aria-label="Fechar detalhes"
          className="p-1 rounded text-fg-subtle hover:text-fg hover:bg-card-2 transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="px-5 py-5">
        <MatchDetail match={match} />
      </div>
    </aside>
  );
}
