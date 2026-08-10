import clsx from 'clsx';

/**
 * The product name plus its accent dot. Shared by the rail and the login card
 * so the two can never drift.
 */
export function Wordmark({ className }: { className?: string }) {
  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <span className="text-[19px] font-extrabold tracking-[-0.02em] text-fg">GOALCAST</span>
      <span className="w-[7px] h-[7px] rounded-full bg-accent shrink-0" />
    </div>
  );
}
