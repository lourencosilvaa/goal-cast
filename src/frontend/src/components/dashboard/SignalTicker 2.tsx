import { ticker } from '@/config/theme';

/**
 * Marquee strip of the day's strongest calls.
 *
 * The list is rendered twice inside the track; the animation translates by
 * exactly -50%, so the second copy lands where the first began and the loop
 * has no visible seam.
 */
export function SignalTicker({ signals }: { signals: string[] }) {
  if (signals.length === 0) return null;

  return (
    <div className="flex items-center h-9 bg-nav border-b border-line overflow-hidden shrink-0">
      <div className="flex items-center gap-2 h-full px-4 shrink-0 border-r border-line bg-nav relative z-10">
        <span className="w-1.5 h-1.5 rounded-full bg-accent pulse-dot" />
        <span className="label-mono text-accent font-semibold">Sinais</span>
      </div>
      <div
        className="marquee-track flex gap-10 pl-6 whitespace-nowrap shrink-0"
        style={{ '--marquee-duration': `${ticker.durationSeconds}s` } as React.CSSProperties}
      >
        {[...signals, ...signals].map((signal, i) => (
          <span key={i} className="font-mono text-xs text-fg-muted">
            {signal}
          </span>
        ))}
      </div>
    </div>
  );
}
