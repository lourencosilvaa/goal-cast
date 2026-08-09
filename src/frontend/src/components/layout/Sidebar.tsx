import { NavLink } from 'react-router-dom';
import clsx from 'clsx';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { NAV_ITEMS } from '@/config/nav';
import { rail } from '@/config/theme';
import { Wordmark } from './Wordmark';

/**
 * Fixed navigation rail.
 *
 * Each entry carries a mono two-letter chip rather than an icon: the codes are
 * unambiguous at a glance and keep the rail on the same typographic system as
 * the data columns it sits beside.
 */
export function Sidebar() {
  const { profile, signOut } = useAuth();

  const items = NAV_ITEMS.filter((item) => !item.adminOnly || profile?.is_admin);

  return (
    <aside
      className="hidden md:flex flex-col shrink-0 bg-nav border-r border-line px-3.5 py-5"
      style={{ width: rail.navWidthPx }}
    >
      <Wordmark className="px-1 pb-2" />

      {profile?.is_admin && (
        <div className="w-fit mx-1 mb-1 px-2 py-0.5 rounded bg-accent/12 border border-accent/45 font-mono text-[10px] font-bold tracking-[0.05em] text-accent">
          ADMIN
        </div>
      )}

      <nav className="flex flex-col gap-1 mt-4">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg text-[13px] font-semibold transition-colors',
                isActive ? 'bg-accent/12 text-fg' : 'text-fg-subtle hover:text-fg hover:bg-card-2',
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={clsx(
                    'w-[30px] h-[22px] shrink-0 inline-flex items-center justify-center rounded border',
                    'font-mono text-[10px] font-bold',
                    isActive ? 'border-accent/45 text-accent' : 'border-line text-fg-subtle',
                  )}
                >
                  {item.code}
                </span>
                <span className="truncate">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="flex-1" />

      <ThemeToggle />

      <button
        onClick={signOut}
        className="mt-1.5 px-2.5 py-2 rounded-lg border border-line bg-transparent text-xs font-semibold text-fg-subtle hover:text-fg hover:border-accent/45 transition-colors cursor-pointer"
      >
        Terminar sessão
      </button>
    </aside>
  );
}
