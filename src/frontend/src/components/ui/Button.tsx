import clsx from 'clsx';
import type { ReactNode, ButtonHTMLAttributes } from 'react';

export type ButtonVariant = 'solid' | 'outline' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  children: ReactNode;
}

const variants: Record<ButtonVariant, string> = {
  /** The one filled control on a screen — reserved for its primary action. */
  solid: 'bg-accent text-bg border border-accent font-bold hover:opacity-90',
  outline:
    'bg-transparent border border-line text-fg-muted hover:text-fg hover:border-accent/45',
  ghost: 'bg-transparent border border-transparent text-fg-subtle hover:text-fg hover:bg-card-2',
  danger:
    'bg-transparent border border-accent-red/40 text-accent-red hover:bg-accent-red/10',
};

const sizes: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-2.5 text-sm',
};

export function Button({
  variant = 'solid',
  size = 'md',
  loading = false,
  children,
  className,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center gap-1.5 rounded-md font-semibold',
        'transition-colors duration-150 cursor-pointer outline-none',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
          <span>A carregar…</span>
        </>
      ) : (
        children
      )}
    </button>
  );
}
