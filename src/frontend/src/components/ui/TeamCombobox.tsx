import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';

/**
 * Shared field styling for every text/select control in the app.
 *
 * Deliberately carries no width: `w-full` and `w-auto` are the same CSS
 * property, so a caller appending one could not reliably override a width
 * baked in here — Tailwind orders the two by its own rules, not by the order
 * they appear in the string. Each call site sets its own.
 */
export const FIELD_CLASS =
  'px-3 py-2.5 rounded-md bg-card border border-line text-fg text-sm ' +
  'placeholder-fg-subtle outline-none focus:border-accent/45 transition-colors';

/** Mono variant, for fields holding keys and other opaque strings. */
export const FIELD_MONO_CLASS = `${FIELD_CLASS} font-mono`;

/** Type-ahead team selector backed by the league's team list. */
export function TeamCombobox({
  label,
  value,
  onChange,
  teams,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  teams: string[];
  placeholder: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  const filtered = teams.filter((team) =>
    team.toLowerCase().includes(query.toLowerCase()),
  );

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  return (
    <div className="space-y-1.5" ref={ref}>
      <label className="label-mono text-fg-subtle block">{label}</label>
      <div className="relative">
        <input
          value={open ? query : value}
          onChange={(e) => {
            setQuery(e.target.value);
            if (!open) setOpen(true);
          }}
          onFocus={() => {
            setOpen(true);
            setQuery('');
          }}
          placeholder={placeholder}
          className={`${FIELD_CLASS} w-full`}
        />
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-subtle pointer-events-none" />
        {open && filtered.length > 0 && (
          <div className="absolute z-50 mt-1 w-full max-h-48 overflow-y-auto rounded-md bg-card border border-line shadow-lg">
            {filtered.map((team) => (
              <button
                key={team}
                type="button"
                className={`w-full text-left px-3 py-2 text-sm hover:bg-card-2 transition-colors ${
                  team === value ? 'text-accent font-semibold' : 'text-fg'
                }`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(team);
                  setQuery('');
                  setOpen(false);
                }}
              >
                {team}
              </button>
            ))}
          </div>
        )}
        {open && filtered.length === 0 && query && (
          <div className="absolute z-50 mt-1 w-full rounded-md bg-card border border-line shadow-lg px-3 py-2.5">
            <p className="text-xs text-fg-subtle">Nenhuma equipa encontrada</p>
          </div>
        )}
      </div>
    </div>
  );
}
