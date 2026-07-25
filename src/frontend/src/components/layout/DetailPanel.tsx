import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import type { MatchPrediction } from '@/types';
import { MatchDetail } from '@/components/match/MatchDetail';

interface DetailPanelProps {
  match: MatchPrediction | null;
  onClose: () => void;
}

/** Docked right-hand panel that shows the full match detail on wide screens. */
export function DetailPanel({ match, onClose }: DetailPanelProps) {
  return (
    <AnimatePresence mode="wait">
      {match && (
        <motion.aside
          key={`${match.home_team}-${match.away_team}`}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 24 }}
          transition={{ duration: 0.2 }}
          className="w-[420px] shrink-0 h-[calc(100vh-3rem)] sticky top-6 glass rounded-2xl border border-line overflow-y-auto scrollbar-hide"
        >
          <div className="flex items-center justify-between px-5 pt-5 pb-3 sticky top-0 z-10 bg-card/80 backdrop-blur">
            <h3 className="text-sm font-semibold text-fg">Detalhes do Jogo</h3>
            <button
              onClick={onClose}
              aria-label="Fechar detalhes"
              className="p-1.5 rounded-lg text-fg-subtle hover:text-fg hover:bg-card-2 transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="px-5 pb-6">
            <MatchDetail match={match} />
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
