import { useCallback, useEffect, useState } from 'react';
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
    <div className="flex items-center gap-3 px-5 md:px-7 py-2.5 bg-accent/12 border-b border-accent/45 text-accent text-[13px] shrink-0">
      <RefreshCw className="w-4 h-4 shrink-0 animate-spin" />
      <span className="flex-1">
        O modelo está a ser re-treinado com dados mais recentes. As previsões podem demorar alguns
        minutos até estarem actualizadas.
      </span>
      <button
        onClick={() => setDismissed(true)}
        className="shrink-0 opacity-60 hover:opacity-100 transition-opacity cursor-pointer"
        aria-label="Fechar"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
