import clsx from 'clsx';
import { useTheme, type ThemePreference } from '@/contexts/ThemeContext';

const LABELS: Record<ThemePreference, string> = {
  light: 'Claro',
  dark: 'Escuro',
  system: 'Sistema',
};

interface ThemeToggleProps {
  /** Switch-only variant, used where there is no room for the label. */
  compact?: boolean;
  className?: string;
}

/**
 * Label + knob switch, matching the design's rail footer.
 *
 * The knob has three resting positions rather than two, because the app keeps
 * a `system` preference the design did not model — dropping it would silently
 * pin every user to whichever theme they last saw.
 */
export function ThemeToggle({ compact = false, className }: ThemeToggleProps) {
  const { preference, cycle } = useTheme();

  /*
   * Resting positions in px, measured against the 44px track and 18px knob.
   * Expressed as `left` for all three so the transition interpolates — a
   * `right`-anchored end state would snap instead of slide.
   */
  const knobOffset: Record<ThemePreference, string> = {
    dark: 'left-[2px]',
    system: 'left-[13px]',
    light: 'left-[24px]',
  };

  return (
    <button
      onClick={cycle}
      title={`Tema: ${LABELS[preference]}`}
      aria-label={`Alternar tema (atual: ${LABELS[preference]})`}
      className={clsx(
        'flex items-center gap-3 cursor-pointer bg-transparent border-0 p-0',
        compact ? '' : 'justify-between w-full px-1 py-2.5',
        className,
      )}
    >
      {!compact && (
        <span className="text-xs font-semibold text-fg-subtle">{LABELS[preference]}</span>
      )}
      <span className="relative block w-11 h-6 rounded-full border border-line bg-card-2 shrink-0">
        <span
          className={clsx(
            'absolute top-0.5 w-[1.125rem] h-[1.125rem] rounded-full bg-accent transition-[left] duration-150',
            knobOffset[preference],
          )}
        />
      </span>
    </button>
  );
}
