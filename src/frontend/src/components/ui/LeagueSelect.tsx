import { LEAGUES } from '@/config/leagues';
import { FIELD_CLASS } from './TeamCombobox';

/** League picker shared by the prediction and statistics pages. */
export function LeagueSelect({
  value,
  onChange,
  label = 'Liga',
}: {
  value: string;
  onChange: (code: string) => void;
  label?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs text-fg-muted font-medium uppercase tracking-wide">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`${FIELD_CLASS} cursor-pointer`}
      >
        {LEAGUES.map((league) => (
          <option key={league.code} value={league.code} className="bg-card text-fg">
            {league.flag} {league.name}
          </option>
        ))}
      </select>
    </div>
  );
}
