import clsx from 'clsx';
import type { ReactNode } from 'react';

/**
 * Uppercase mono micro-heading. The design uses these instead of bold sans
 * headings so that chrome never competes with the figures it introduces.
 */
export function SectionLabel({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={clsx('label-mono text-fg-subtle', className)}>{children}</div>;
}
