import clsx from 'clsx';
import type { ReactNode } from 'react';

/**
 * Standard scrolling body for pages that are a single column.
 *
 * The shell is fixed-height, so a page that does not claim its own scroll
 * region simply gets clipped — this is that claim, in one place.
 */
export function PageBody({
  children,
  className,
  intro,
  maxWidth = 'full',
}: {
  children: ReactNode;
  className?: string;
  /** One-line explanation under the top bar, as in the design. */
  intro?: string;
  maxWidth?: 'full' | 'md' | 'lg';
}) {
  const widthClass = {
    full: '',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
  }[maxWidth];

  return (
    <div className="flex-1 overflow-y-auto px-5 md:px-7 py-6">
      <div className={clsx(widthClass, className)}>
        {intro && <p className="text-[13px] text-fg-muted mb-5 max-w-[640px]">{intro}</p>}
        {children}
      </div>
    </div>
  );
}
