import { NavLink } from 'react-router-dom';
import { BarChart3, Settings, TrendingUp, ShieldCheck, LogOut, Wand2, Shield } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/value-bets', label: 'Value Bets', icon: TrendingUp },
  { to: '/custom-predict', label: 'Prever', icon: Wand2 },
  { to: '/team-stats', label: 'Equipas', icon: Shield },
  { to: '/settings', label: 'Definições', icon: Settings },
];

export function MobileNav() {
  const { profile, signOut } = useAuth();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 glass border-t border-line"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      <div className="flex justify-around py-2 items-center">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] transition-all ${
                isActive ? 'text-accent-blue' : 'text-fg-subtle'
              }`
            }
          >
            <item.icon className="w-5 h-5" />
            {item.label}
          </NavLink>
        ))}

        {profile?.is_admin && (
          <NavLink
            to="/admin"
            className={({ isActive }) =>
              `flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] transition-all ${
                isActive ? 'text-accent-purple' : 'text-fg-subtle'
              }`
            }
          >
            <ShieldCheck className="w-5 h-5" />
            Admin
          </NavLink>
        )}

        <ThemeToggle compact />

        <button
          onClick={signOut}
          className="flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] transition-all text-fg-subtle hover:text-accent-red"
        >
          <LogOut className="w-5 h-5" />
          Sair
        </button>
      </div>
    </nav>
  );
}
