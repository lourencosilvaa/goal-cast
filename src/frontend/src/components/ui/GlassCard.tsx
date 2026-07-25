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
}: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { scale: 1.01, y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className={clsx(
        'glass rounded-2xl border border-line overflow-hidden relative',
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
