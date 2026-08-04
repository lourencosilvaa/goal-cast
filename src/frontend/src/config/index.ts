export const config = {
  apiUrl: import.meta.env.VITE_API_URL || '',
  supabaseUrl: import.meta.env.VITE_SUPABASE_URL || '',
  supabaseAnonKey: import.meta.env.VITE_SUPABASE_ANON_KEY || '',
};

/**
 * Local-only login bypass. `import.meta.env.DEV` is compiled to `false` by
 * `vite build` (the production Docker image), so this can NEVER be true in a
 * deployed build regardless of env vars. Opt in locally by setting
 * `VITE_DEV_AUTH_BYPASS=true` in `src/frontend/.env.local`.
 */
export const devAuth = {
  enabled: import.meta.env.DEV && import.meta.env.VITE_DEV_AUTH_BYPASS === 'true',
  /** Sentinel token the backend accepts when its own DEV_AUTH_BYPASS is on. */
  token: 'dev-bypass',
  user: {
    user_id: import.meta.env.VITE_DEV_USER_ID || 'dev-user',
    email: import.meta.env.VITE_DEV_USER_EMAIL || 'dev@localhost',
  },
} as const;

/**
 * Layout constants — the master/detail breakpoint decides whether a selected
 * match opens in a docked right panel (wide screens) or a dedicated sub-page
 * (narrow screens).
 */
export const layout = {
  /** Minimum viewport width (px) that shows the docked detail panel. */
  detailBreakpointPx: 1280,
  /** Matching media query string for `useMediaQuery`. */
  detailPanelQuery: '(min-width: 1280px)',
} as const;

/** Persistence key for the selected theme preference. */
export const THEME_STORAGE_KEY = 'goalcast-theme';
