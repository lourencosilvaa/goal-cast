import { motion } from 'framer-motion';
import clsx from 'clsx';
import type { ReactNode } from 'react';

export type Accent = 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'neutral';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  /** Semantic tint applied as a subtle corner gradient. */
  accent?: Accent;
  padding?: 'sm' | 'md' | 'lg' | 'none';
  hover?: boolean;
  /**
   * Clipping behaviour. The default `hidden` keeps the rounded corners crisp,
   * but it also clips absolutely-positioned children — a card containing a
   * dropdown must opt into `visible` or the list is cut off at the card edge.
   *
   * Set as a swap rather than an appended class: without `tailwind-merge`,
   * `overflow-hidden overflow-visible` is resolved by stylesheet order, not by
   * the order written here.
   */
  overflow?: 'hidden' | 'visible';
}

const accentOverlay: Record<Accent, string> = {
  green: 'before:from-accent-green/10',
  red: 'before:from-accent-red/10',
  blue: 'before:from-accent-blue/10',
  amber: 'before:from-accent-amber/10',
  purple: 'before:from-accent-purple/10',
  neutral: 'before:from-fg/5',
};

const paddingMap: Record<string, string> = {
  sm: 'p-3',
  md: 'p-5',
  lg: 'p-7',
  none: 'p-0',
};

export function GlassCard({
  children,
  className,
  accent,
  padding = 'md',
  hover = false,
  overflow = 'hidden',
}: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { scale: 1.01, y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className={clsx(
        'glass rounded-2xl border border-line relative',
        overflow === 'visible' ? 'overflow-visible' : 'overflow-hidden',
        paddingMap[padding],
        accent &&
          `before:absolute before:inset-0 before:bg-gradient-to-br ${accentOverlay[accent]} before:to-transparent before:pointer-events-none`,
        className,
      )}
    >
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}
