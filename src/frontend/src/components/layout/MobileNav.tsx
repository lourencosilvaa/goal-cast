import { NavLink } from 'react-router-dom';
import clsx from 'clsx';
import { LogOut } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { NAV_ITEMS } from '@/config/nav';

/** The rail's counterpart below the desktop breakpoint. */
export function MobileNav() {
  const { profile, signOut } = useAuth();

  const items = NAV_ITEMS.filter((item) => !item.adminOnly || profile?.is_admin);

  return (
    <nav
      className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-nav border-t border-line"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      <div className="flex justify-around items-center py-2 gap-1">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex flex-col items-center gap-1 px-2 py-1 transition-colors',
                isActive ? 'text-accent' : 'text-fg-subtle',
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={clsx(
                    'w-7 h-5 inline-flex items-center justify-center rounded border font-mono text-[10px] font-bold',
                    isActive ? 'border-accent/45' : 'border-line',
                  )}
                >
                  {item.code}
                </span>
                <span className="text-[9px] font-semibold">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}

        <div className="flex flex-col items-center gap-1 px-2 py-1">
          <ThemeToggle compact />
          <span className="text-[9px] font-semibold text-fg-subtle">Tema</span>
        </div>

        <button
          onClick={signOut}
          className="flex flex-col items-center gap-1 px-2 py-1 text-fg-subtle hover:text-accent-red transition-colors cursor-pointer"
        >
          <LogOut className="w-4 h-4" />
          <span className="text-[9px] font-semibold">Sair</span>
        </button>
      </div>
    </nav>
  );
}
