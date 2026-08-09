import clsx from 'clsx';
import type { ReactNode } from 'react';

export type Accent = 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'accent' | 'neutral';

interface PanelProps {
  children: ReactNode;
  className?: string;
  /** Tints the border only — the surface itself stays flat. */
  accent?: Accent;
  padding?: 'sm' | 'md' | 'lg' | 'none';
  /**
   * Clipping behaviour. The default `hidden` keeps the corners crisp, but it
   * also clips absolutely-positioned children — a panel containing a dropdown
   * must opt into `visible` or the list is cut off at the panel edge.
   */
  overflow?: 'hidden' | 'visible';
}

const accentBorder: Record<Accent, string> = {
  green: 'border-accent-green/40',
  red: 'border-accent-red/40',
  blue: 'border-accent-blue/40',
  amber: 'border-accent-amber/40',
  purple: 'border-accent-purple/40',
  accent: 'border-accent/45',
  neutral: 'border-line-soft',
};

const paddingMap: Record<string, string> = {
  sm: 'p-3',
  md: 'p-5',
  lg: 'p-7',
  none: 'p-0',
};

/**
 * The flat surface every page composes from: one background step above the
 * page, a single hairline border, small radius. No blur, no gradient — depth
 * in this design comes from borders and density, not from glass.
 */
export function Panel({
  children,
  className,
  accent = 'neutral',
  padding = 'md',
  overflow = 'hidden',
}: PanelProps) {
  return (
    <div
      className={clsx(
        'rounded-lg border bg-card',
        accentBorder[accent],
        overflow === 'visible' ? 'overflow-visible' : 'overflow-hidden',
        paddingMap[padding],
        className,
      )}
    >
      {children}
    </div>
  );
}
