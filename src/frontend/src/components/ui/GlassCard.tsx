import { motion } from 'framer-motion';
import clsx from 'clsx';
import type { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  gradient?: 'purple' | 'blue' | 'teal' | 'green' | 'pink';
  padding?: 'sm' | 'md' | 'lg' | 'none';
  hover?: boolean;
}

const gradientMap: Record<string, string> = {
  purple: 'before:from-violet-600/10',
  blue: 'before:from-blue-600/10',
  teal: 'before:from-teal-600/10',
  green: 'before:from-emerald-600/10',
  pink: 'before:from-pink-600/10',
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
  gradient,
  padding = 'md',
  hover = false,
}: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { scale: 1.01, y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className={clsx(
        'glass rounded-2xl border border-white/[0.06] overflow-hidden relative',
        paddingMap[padding],
        gradient &&
          `before:absolute before:inset-0 before:bg-gradient-to-br ${gradientMap[gradient]} before:to-transparent before:pointer-events-none`,
        className,
      )}
    >
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}
