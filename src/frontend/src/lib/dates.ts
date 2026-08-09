/**
 * Report-date helpers.
 *
 * The API speaks `dd/mm/yyyy` throughout, so the app does too; these keep the
 * parsing and the Portuguese labelling in one place rather than repeated in
 * every page that carries a date picker.
 */

const MONTHS_PT = [
  'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
  'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez',
];

const WEEKDAYS_PT = ['DOM', 'SEG', 'TER', 'QUA', 'QUI', 'SEX', 'SAB'];

/** Today in the API's `dd/mm/yyyy` form. */
export function todayStr(): string {
  const now = new Date();
  const day = String(now.getDate()).padStart(2, '0');
  const month = String(now.getMonth() + 1).padStart(2, '0');
  return `${day}/${month}/${now.getFullYear()}`;
}

/** Parses `dd/mm/yyyy` into a Date, or null when the shape does not match. */
export function parseReportDate(value: string): Date | null {
  const [day, month, year] = value.split('/').map(Number);
  if (!day || !month || !year) return null;
  return new Date(year, month - 1, day);
}

/** `09/08/2026` → `9 Ago`. */
export function formatDateLabel(value: string): string {
  const parsed = parseReportDate(value);
  if (!parsed) return value;
  return `${parsed.getDate()} ${MONTHS_PT[parsed.getMonth()]}`;
}

/** `09/08/2026` → `SAB 09`, the compact form used on the dashboard pills. */
export function formatDayPill(value: string): string {
  const parsed = parseReportDate(value);
  if (!parsed) return value;
  return `${WEEKDAYS_PT[parsed.getDay()]} ${String(parsed.getDate()).padStart(2, '0')}`;
}

/** Label with today spelled out, for selects and prose. */
export function describeDate(value: string): string {
  return value === todayStr() ? `Hoje (${formatDateLabel(value)})` : formatDateLabel(value);
}

/**
 * Today plus every date the API knows about, most recent first.
 *
 * Today is always included even when it has no fixtures, so the picker never
 * silently jumps the user to another day.
 */
export function withToday(availableDates: string[]): string[] {
  const today = todayStr();
  return [today, ...availableDates.filter((date) => date !== today)].sort((a, b) => {
    const left = parseReportDate(a)?.getTime() ?? 0;
    const right = parseReportDate(b)?.getTime() ?? 0;
    return right - left;
  });
}
