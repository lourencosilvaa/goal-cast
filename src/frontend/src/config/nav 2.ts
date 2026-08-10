/**
 * The application's navigation, defined once.
 *
 * The rail, the mobile bar and the top bar's page title all read from this
 * list, so a route can never appear in one and be missing from another.
 */
export interface NavItem {
  to: string;
  /** Two-letter mono code shown in the rail chip. */
  code: string;
  label: string;
  /** Only rendered for users whose profile carries `is_admin`. */
  adminOnly?: boolean;
}

export const NAV_ITEMS: readonly NavItem[] = [
  { to: '/', code: 'DB', label: 'Dashboard' },
  { to: '/value-bets', code: 'VB', label: 'Value Bets' },
  { to: '/custom-predict', code: 'PJ', label: 'Prever Jogo' },
  { to: '/team-stats', code: 'EQ', label: 'Equipas' },
  { to: '/settings', code: 'DF', label: 'Definições' },
  { to: '/admin', code: 'AD', label: 'Admin', adminOnly: true },
] as const;

/** Titles for routes that are reachable but not part of the rail. */
const SUBPAGE_TITLES: Record<string, string> = {
  '/match': 'Detalhe do Jogo',
};

/** Resolves the top-bar title for a pathname. */
export function titleForPath(pathname: string): string {
  const exact = NAV_ITEMS.find((item) => item.to === pathname);
  if (exact) return exact.label;

  const subpage = Object.entries(SUBPAGE_TITLES).find(([prefix]) =>
    pathname.startsWith(prefix),
  );
  if (subpage) return subpage[1];

  // Longest non-root prefix wins, so nested routes inherit their section name.
  const nested = NAV_ITEMS.filter((item) => item.to !== '/' && pathname.startsWith(item.to)).sort(
    (a, b) => b.to.length - a.to.length,
  )[0];
  return nested?.label ?? NAV_ITEMS[0].label;
}
