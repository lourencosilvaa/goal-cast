import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, X } from 'lucide-react';
import { fetchRetrainStatus } from '@/lib/api';

const POLL_INTERVAL_MS = 30_000;

export function RetrainingBanner() {
  const [retraining, setRetraining] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const poll = useCallback(async () => {
    const active = await fetchRetrainStatus();
    setRetraining(active);
    if (active) setDismissed(false);
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [poll]);

  if (!retraining || dismissed) return null;

  return (
    <div className="flex items-center gap-3 px-4 py-2.5 bg-amber-500/10 border-b border-amber-500/20 text-amber-300 text-sm">
      <RefreshCw className="w-4 h-4 shrink-0 animate-spin" />
      <span className="flex-1">
        O modelo está a ser re-treinado com dados mais recentes. As previsões podem demorar alguns minutos até estarem actualizadas.
      </span>
      <button
        onClick={() => setDismissed(true)}
        className="shrink-0 text-amber-400/60 hover:text-amber-300 transition-colors"
        aria-label="Fechar"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
