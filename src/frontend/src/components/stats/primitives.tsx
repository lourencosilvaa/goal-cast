import { motion } from 'framer-motion';
import type { Accent } from './format';


const ACCENT_SURFACE: Record<Accent, string> = {
  green: 'bg-accent-green/10 border-accent-green/25',
  blue: 'bg-accent-blue/10 border-accent-blue/25',
  amber: 'bg-accent-amber/10 border-accent-amber/25',
  red: 'bg-accent-red/10 border-accent-red/25',
  purple: 'bg-accent-purple/10 border-accent-purple/25',
};

/** Compact labelled figure. The workhorse of every statistics grid. */
export function StatTile({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: Accent;
}) {
  return (
    <div
      className={`text-center px-2 py-2 rounded-md border ${
        accent ? ACCENT_SURFACE[accent] : 'bg-card-2 border-line'
      }`}
    >
      <p className="text-[10px] text-fg-subtle mb-0.5 truncate">{label}</p>
      <p className="text-sm font-semibold text-fg">{value}</p>
      {hint && <p className="text-[10px] text-fg-subtle mt-0.5">{hint}</p>}
    </div>
  );
}

/** Titled block with an optional leading icon. */
export function StatSection({
  icon,
  title,
  subtitle,
  children,
}: {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h4 className="text-xs text-fg-subtle uppercase tracking-wider flex items-center gap-1.5">
          {icon}
          {title}
        </h4>
        {subtitle && <span className="text-[10px] text-fg-subtle">{subtitle}</span>}
      </div>
      {children}
    </div>
  );
}

/** Result letters (W/D/L) as coloured pills, most recent first. */
export function FormPills({ sequence }: { sequence: string[] }) {
  if (sequence.length === 0) {
    return <p className="text-xs text-fg-subtle">Sem jogos registados</p>;
  }
  const style: Record<string, string> = {
    W: 'bg-accent-green/15 text-accent-green border-accent-green/25',
    D: 'bg-accent-amber/15 text-accent-amber border-accent-amber/25',
    L: 'bg-accent-red/15 text-accent-red border-accent-red/25',
  };
  const label: Record<string, string> = { W: 'V', D: 'E', L: 'D' };
  return (
    <div className="flex gap-1.5">
      {sequence.map((result, i) => (
        <span
          key={i}
          className={`w-6 h-6 rounded-md border text-[11px] font-semibold flex items-center justify-center ${
            style[result] ?? 'bg-card-2 text-fg-muted border-line'
          }`}
          title={result}
        >
          {label[result] ?? result}
        </span>
      ))}
    </div>
  );
}

/** Stacked win/draw/loss proportion bar with the raw counts underneath. */
export function RecordBar({
  wins,
  draws,
  losses,
  labels = ['Vitórias', 'Empates', 'Derrotas'],
}: {
  wins: number;
  draws: number;
  losses: number;
  labels?: [string, string, string] | string[];
}) {
  const total = wins + draws + losses;
  const share = (value: number) => (total > 0 ? (value / total) * 100 : 0);
  const segments = [
    { value: wins, color: 'bg-accent-green', label: labels[0] },
    { value: draws, color: 'bg-accent-amber', label: labels[1] },
    { value: losses, color: 'bg-accent-red', label: labels[2] },
  ];

  return (
    <div className="space-y-2">
      <div className="flex h-2.5 rounded-full overflow-hidden bg-card-2">
        {segments.map((segment, i) => (
          <motion.div
            key={i}
            className={segment.color}
            initial={{ width: 0 }}
            animate={{ width: `${share(segment.value)}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2">
        {segments.map((segment, i) => (
          <div key={i} className="text-center">
            <p className="text-sm font-semibold text-fg">{segment.value}</p>
            <p className="text-[10px] text-fg-subtle">{segment.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Two values facing each other across a shared label. */
export function ComparisonRow({
  label,
  homeValue,
  awayValue,
  highlight = 'higher',
}: {
  label: string;
  homeValue: number;
  awayValue: number;
  highlight?: 'higher' | 'lower' | 'none';
}) {
  const homeWins =
    highlight === 'none'
      ? false
      : highlight === 'higher'
        ? homeValue > awayValue
        : homeValue < awayValue;
  const awayWins =
    highlight === 'none'
      ? false
      : highlight === 'higher'
        ? awayValue > homeValue
        : awayValue < homeValue;
  const total = Math.abs(homeValue) + Math.abs(awayValue);
  const homeShare = total > 0 ? (Math.abs(homeValue) / total) * 100 : 50;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className={homeWins ? 'text-accent-green font-semibold' : 'text-fg-muted'}>
          {homeValue.toFixed(2)}
        </span>
        <span className="text-[10px] text-fg-subtle uppercase tracking-wide">{label}</span>
        <span className={awayWins ? 'text-accent-blue font-semibold' : 'text-fg-muted'}>
          {awayValue.toFixed(2)}
        </span>
      </div>
      <div className="flex h-1.5 rounded-full overflow-hidden bg-card-2">
        <motion.div
          className="bg-accent-green/70"
          initial={{ width: 0 }}
          animate={{ width: `${homeShare}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
        <div className="flex-1 bg-accent-blue/70" />
      </div>
    </div>
  );
}
