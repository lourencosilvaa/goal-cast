import { Moon, Sun, Monitor } from 'lucide-react';
import clsx from 'clsx';
import { useTheme, type ThemePreference } from '@/contexts/ThemeContext';

const ICONS: Record<ThemePreference, typeof Sun> = {
  light: Sun,
  dark: Moon,
  system: Monitor,
};

const LABELS: Record<ThemePreference, string> = {
  light: 'Claro',
  dark: 'Escuro',
  system: 'Sistema',
};

interface ThemeToggleProps {
  /** Compact icon-only variant (used in the mobile nav). */
  compact?: boolean;
  className?: string;
}

export function ThemeToggle({ compact = false, className }: ThemeToggleProps) {
  const { preference, cycle } = useTheme();
  const Icon = ICONS[preference];

  return (
    <button
      onClick={cycle}
      title={`Tema: ${LABELS[preference]}`}
      aria-label={`Alternar tema (atual: ${LABELS[preference]})`}
      className={clsx(
        'flex items-center gap-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer',
        'text-fg-muted hover:text-fg hover:bg-card-2 border border-transparent',
        compact ? 'p-1.5' : 'px-3 py-2.5 w-full',
        className,
      )}
    >
      <Icon className="w-4 h-4" />
      {!compact && <span>{LABELS[preference]}</span>}
    </button>
  );
}
