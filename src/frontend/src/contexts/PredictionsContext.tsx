import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { fetchAllPredictions, refreshPredictions } from '@/lib/api';
import type { LeaguePredictions } from '@/types';

interface DateCache {
  data: LeaguePredictions[];
  loading: boolean;
  error: string | null;
}

interface PredictionsContextValue {
  cache: Record<string, DateCache>;
  fetchForDate: (date: string) => void;
  invalidateDate: (date: string) => Promise<void>;
}

const PredictionsContext = createContext<PredictionsContextValue | null>(null);

export function PredictionsProvider({ children }: { children: React.ReactNode }) {
  const [cache, setCache] = useState<Record<string, DateCache>>({});
  const inFlightRef = useRef<Set<string>>(new Set());

  const fetchForDate = useCallback((date: string) => {
    if (inFlightRef.current.has(date)) return;
    setCache((prev) => {
      // Already have fresh data — skip
      if (prev[date]?.data.length && !prev[date].error) return prev;
      return { ...prev, [date]: { data: prev[date]?.data ?? [], loading: true, error: null } };
    });
    inFlightRef.current.add(date);
    fetchAllPredictions(date)
      .then((data) => {
        setCache((prev) => ({ ...prev, [date]: { data, loading: false, error: null } }));
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : 'Failed to load predictions';
        setCache((prev) => ({
          ...prev,
          [date]: { data: prev[date]?.data ?? [], loading: false, error: msg },
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(date);
      });
  }, []);

  const invalidateDate = useCallback(async (date: string) => {
    await refreshPredictions();
    setCache((prev) => {
      const next = { ...prev };
      delete next[date];
      return next;
    });
    fetchForDate(date);
  }, [fetchForDate]);

  return (
    <PredictionsContext.Provider value={{ cache, fetchForDate, invalidateDate }}>
      {children}
    </PredictionsContext.Provider>
  );
}

export function usePredictions(date: string) {
  const ctx = useContext(PredictionsContext);
  if (!ctx) throw new Error('usePredictions must be used within PredictionsProvider');
  const { cache, fetchForDate, invalidateDate } = ctx;

  useEffect(() => {
    fetchForDate(date);
  }, [date, fetchForDate]);

  const state: DateCache = cache[date] ?? { data: [], loading: true, error: null };
  const invalidate = useCallback(() => invalidateDate(date), [invalidateDate, date]);

  return { ...state, invalidate };
}
