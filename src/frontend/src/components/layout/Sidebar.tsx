import { NavLink } from 'react-router-dom';
import { BarChart3, Settings, Radio, TrendingUp, ShieldCheck, LogOut, Wand2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/value-bets', label: 'Value Bets', icon: TrendingUp },
  { to: '/custom-predict', label: 'Prever Jogo', icon: Wand2 },
  { to: '/settings', label: 'Definições', icon: Settings },
];

export function Sidebar() {
  const { profile, signOut } = useAuth();

  return (
    <aside className="hidden md:flex flex-col w-64 h-screen glass border-r border-line p-4">
      {/* Logo */}
      <div className="flex items-center gap-3 px-2 mb-8">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center">
          <Radio className="w-5 h-5 text-white" />
        </div>
        <span className="text-lg font-bold gradient-text">GoalCast</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-1 flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-accent-blue/12 border border-accent-blue/25 text-fg'
                  : 'text-fg-muted hover:text-fg hover:bg-card-2 border border-transparent'
              }`
            }
          >
            <item.icon className="w-4.5 h-4.5" />
            {item.label}
          </NavLink>
        ))}

        {profile?.is_admin && (
          <NavLink
            to="/admin"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-accent-purple/12 border border-accent-purple/25 text-fg'
                  : 'text-fg-muted hover:text-fg hover:bg-card-2 border border-transparent'
              }`
            }
          >
            <ShieldCheck className="w-4.5 h-4.5" />
            Admin
          </NavLink>
        )}
      </nav>

      {/* Theme + Logout */}
      <div className="flex flex-col gap-1 border-t border-line pt-2">
        <ThemeToggle />
        <button
          onClick={signOut}
          className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 text-fg-muted hover:text-accent-red hover:bg-accent-red/10 border border-transparent w-full"
        >
          <LogOut className="w-4 h-4" />
          Terminar sessão
        </button>
      </div>
    </aside>
  );
}
