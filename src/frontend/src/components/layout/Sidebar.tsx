import { NavLink } from 'react-router-dom';
import { BarChart3, Settings, Trophy, TrendingUp, ShieldCheck } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/value-bets', label: 'Value Bets', icon: TrendingUp },
  { to: '/settings', label: 'Definições', icon: Settings },
];

export function Sidebar() {
  const { profile } = useAuth();

  return (
    <aside className="hidden md:flex flex-col w-64 h-screen glass border-r border-white/[0.06] p-4">
      {/* Logo */}
      <div className="flex items-center gap-3 px-2 mb-8">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
          <Trophy className="w-5 h-5 text-white" />
        </div>
        <span className="text-lg font-bold gradient-text">FootballAI</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-1 flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-green-600/20 to-emerald-600/10 border border-green-500/20 text-white'
                  : 'text-white/50 hover:text-white/80 hover:bg-white/5 border border-transparent'
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
                  ? 'bg-gradient-to-r from-purple-600/20 to-purple-600/10 border border-purple-500/20 text-white'
                  : 'text-white/50 hover:text-white/80 hover:bg-white/5 border border-transparent'
              }`
            }
          >
            <ShieldCheck className="w-4.5 h-4.5" />
            Admin
          </NavLink>
        )}
      </nav>
    </aside>
  );
}
