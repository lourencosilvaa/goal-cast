import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';

/** Shared field styling for the pickers on the prediction/statistics pages. */
export const FIELD_CLASS =
  'w-full px-4 py-3 rounded-xl bg-card-2 border border-line text-fg placeholder-fg-subtle focus:outline-none focus:border-accent-blue/40 transition-colors text-sm';

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
      <label className="text-xs text-fg-muted font-medium uppercase tracking-wide">
        {label}
      </label>
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
          className={FIELD_CLASS}
        />
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-subtle pointer-events-none" />
        {open && filtered.length > 0 && (
          <div className="absolute z-50 mt-1 w-full max-h-48 overflow-y-auto rounded-xl bg-card border border-line shadow-xl">
            {filtered.map((team) => (
              <button
                key={team}
                type="button"
                className={`w-full text-left px-4 py-2 text-sm hover:bg-card-2 transition-colors ${
                  team === value ? 'text-accent-green' : 'text-fg'
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
          <div className="absolute z-50 mt-1 w-full rounded-xl bg-card border border-line shadow-xl px-4 py-3">
            <p className="text-xs text-fg-subtle">Nenhuma equipa encontrada</p>
          </div>
        )}
      </div>
    </div>
  );
}
