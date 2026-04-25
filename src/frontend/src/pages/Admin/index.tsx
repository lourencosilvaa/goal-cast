import { useState, useEffect, useCallback } from 'react';
import { Users, UserPlus, ShieldCheck, ShieldOff, Loader2, Key } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import {
  adminListUsers,
  adminCreateUser,
  adminSetApproved,
  type AdminUser,
} from '@/lib/api';

function StatusBadge({ approved, isAdmin }: { approved: boolean; isAdmin: boolean }) {
  if (isAdmin) {
    return (
      <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/20 font-medium">
        Admin
      </span>
    );
  }
  return approved ? (
    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 font-medium">
      Activo
    </span>
  ) : (
    <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-300 border border-red-500/20 font-medium">
      Revogado
    </span>
  );
}

export function AdminPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showCreate, setShowCreate] = useState(false);
  const [newEmail, setNewEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const [toggling, setToggling] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setUsers(await adminListUsers());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro ao carregar utilizadores');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreateError(null);
    setCreating(true);
    try {
      const created = await adminCreateUser(newEmail.trim(), newPassword);
      setUsers((prev) => [...prev, created]);
      setNewEmail('');
      setNewPassword('');
      setShowCreate(false);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Erro ao criar utilizador');
    } finally {
      setCreating(false);
    }
  }

  async function handleToggle(user: AdminUser) {
    setToggling(user.user_id);
    try {
      await adminSetApproved(user.user_id, !user.approved);
      setUsers((prev) =>
        prev.map((u) =>
          u.user_id === user.user_id ? { ...u, approved: !u.approved } : u,
        ),
      );
    } finally {
      setToggling(null);
    }
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white/90">Admin</h1>
          <p className="text-sm text-white/40 mt-1">Gestão de utilizadores</p>
        </div>
        <NeonButton onClick={() => setShowCreate((v) => !v)}>
          <UserPlus className="w-4 h-4 mr-1.5" />
          Novo utilizador
        </NeonButton>
      </div>

      {/* Create user form */}
      {showCreate && (
        <GlassCard gradient="green" className="mb-6 max-w-md">
          <h2 className="text-sm font-semibold text-white/80 mb-4 flex items-center gap-2">
            <UserPlus className="w-4 h-4" /> Criar utilizador
          </h2>
          <form onSubmit={handleCreate} className="space-y-3">
            <div>
              <label className="block text-xs text-white/50 mb-1">Email</label>
              <input
                type="email"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
                required
                placeholder="email@exemplo.com"
                className="w-full px-3 py-2 rounded-xl glass border border-white/10 focus:border-green-500/40 focus:outline-none text-sm text-white/80 placeholder:text-white/20 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-white/50 mb-1">
                <Key className="w-3 h-3 inline mr-1" />
                Password temporária
              </label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                minLength={6}
                placeholder="••••••••"
                className="w-full px-3 py-2 rounded-xl glass border border-white/10 focus:border-green-500/40 focus:outline-none text-sm text-white/80 placeholder:text-white/20 transition-colors"
              />
            </div>
            {createError && (
              <p className="text-xs text-red-400">{createError}</p>
            )}
            <div className="flex gap-2 pt-1">
              <NeonButton type="submit" disabled={creating} className="flex-1">
                {creating ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : 'Criar'}
              </NeonButton>
              <NeonButton variant="ghost" type="button" onClick={() => setShowCreate(false)}>
                Cancelar
              </NeonButton>
            </div>
          </form>
        </GlassCard>
      )}

      {/* Users table */}
      <GlassCard gradient="green">
        <div className="flex items-center gap-2 mb-5">
          <Users className="w-5 h-5 text-green-400" />
          <h2 className="text-sm font-semibold text-white/80">
            Utilizadores ({users.length})
          </h2>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 text-green-400 animate-spin" />
          </div>
        )}

        {error && <p className="text-sm text-red-400 text-center py-4">{error}</p>}

        {!loading && !error && users.length === 0 && (
          <p className="text-sm text-white/30 text-center py-4">
            Nenhum utilizador encontrado.
          </p>
        )}

        {!loading && users.length > 0 && (
          <div className="space-y-2">
            {users.map((user) => (
              <div
                key={user.user_id}
                className="flex items-center justify-between py-3 px-4 rounded-xl bg-white/[0.03] border border-white/[0.06]"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500/30 to-emerald-600/30 flex items-center justify-center shrink-0">
                    <span className="text-xs font-bold text-green-300 uppercase">
                      {user.email[0]}
                    </span>
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm text-white/80 font-mono truncate">{user.email}</p>
                    <p className="text-[10px] text-white/30 font-mono">{user.user_id.slice(0, 8)}…</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0 ml-4">
                  <StatusBadge approved={user.approved} isAdmin={user.is_admin} />
                  {!user.is_admin && (
                    <button
                      onClick={() => handleToggle(user)}
                      disabled={toggling === user.user_id}
                      className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border transition-colors disabled:opacity-40"
                      style={
                        user.approved
                          ? { borderColor: 'rgba(239,68,68,0.3)', color: 'rgb(252,165,165)' }
                          : { borderColor: 'rgba(34,197,94,0.3)', color: 'rgb(134,239,172)' }
                      }
                    >
                      {toggling === user.user_id ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : user.approved ? (
                        <><ShieldOff className="w-3 h-3" /> Revogar</>
                      ) : (
                        <><ShieldCheck className="w-3 h-3" /> Aprovar</>
                      )}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </GlassCard>
    </div>
  );
}
