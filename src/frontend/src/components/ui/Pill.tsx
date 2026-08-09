import clsx from 'clsx';
import type { ReactNode } from 'react';

/**
 * Toggle chip. Active state is an accent-tinted fill with an accent border —
 * the same treatment used by the nav rail, so "selected" reads identically
 * wherever it appears.
 */
export function Pill({
  active,
  onClick,
  children,
  mono = false,
  title,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
  /** Mono type, for dates and other figures. */
  mono?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={clsx(
        'px-3 py-1.5 rounded-md border text-xs font-semibold whitespace-nowrap',
        'transition-colors cursor-pointer outline-none',
        mono && 'font-mono',
        active
          ? 'bg-accent/12 border-accent/45 text-fg'
          : 'bg-transparent border-line text-fg-subtle hover:text-fg',
      )}
    >
      {children}
    </button>
  );
}
