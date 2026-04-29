import { NavLink } from 'react-router-dom';
import { BarChart3, Settings, TrendingUp, ShieldCheck, LogOut, Wand2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/value-bets', label: 'Value Bets', icon: TrendingUp },
  { to: '/custom-predict', label: 'Prever', icon: Wand2 },
  { to: '/settings', label: 'Definições', icon: Settings },
];

export function MobileNav() {
  const { profile, signOut } = useAuth();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 glass border-t border-white/[0.06]"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      <div className="flex justify-around py-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] transition-all ${
                isActive
                  ? 'text-green-400 drop-shadow-[0_0_6px_rgba(34,197,94,0.5)]'
                  : 'text-white/40'
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
                isActive
                  ? 'text-purple-400 drop-shadow-[0_0_6px_rgba(168,85,247,0.5)]'
                  : 'text-white/40'
              }`
            }
          >
            <ShieldCheck className="w-5 h-5" />
            Admin
          </NavLink>
        )}

        <button
          onClick={signOut}
          className="flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] transition-all text-white/40 hover:text-red-400"
        >
          <LogOut className="w-5 h-5" />
          Sair
        </button>
      </div>
    </nav>
  );
}
