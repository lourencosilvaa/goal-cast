import { useCallback, useEffect, useState } from 'react';
import { Loader2, UserPlus } from 'lucide-react';
import clsx from 'clsx';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { Pill } from '@/components/ui/Pill';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { PageBody } from '@/components/layout/PageBody';
import { FIELD_CLASS } from '@/components/ui/TeamCombobox';
import { TeamAliasPanel } from '@/components/admin/TeamAliasPanel';
import {
  adminCreateUser,
  adminListUsers,
  adminSetApproved,
  type AdminUser,
} from '@/lib/api';

type Tab = 'users' | 'aliases';

const TABS: { key: Tab; label: string }[] = [
  { key: 'users', label: 'Utilizadores' },
  { key: 'aliases', label: 'Nomes de Equipas' },
];

function StatusBadge({ approved, isAdmin }: { approved: boolean; isAdmin: boolean }) {
  const [label, tone] = isAdmin
    ? ['ADMIN', 'text-accent border-accent/45']
    : approved
      ? ['ATIVO', 'text-accent-green border-accent-green/40']
      : ['REVOGADO', 'text-accent-red border-accent-red/40'];

  return (
    <span
      className={clsx(
        'shrink-0 px-2 py-0.5 rounded border font-mono text-[10px] font-bold',
        tone,
      )}
    >
      {label}
    </span>
  );
}

function UsersTab() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    void load();
  }, [load]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreateError(null);
    setCreating(true);
    try {
      const created = await adminCreateUser(newEmail.trim(), newPassword);
      setUsers((prev) => [...prev, created]);
      setNewEmail('');
      setNewPassword('');
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
        prev.map((u) => (u.user_id === user.user_id ? { ...u, approved: !u.approved } : u)),
      );
    } finally {
      setToggling(null);
    }
  }

  return (
    <>
      <form onSubmit={handleCreate} className="flex items-end gap-3 flex-wrap mb-2">
        <input
          type="email"
          value={newEmail}
          onChange={(e) => setNewEmail(e.target.value)}
          required
          placeholder="Email"
          className={`${FIELD_CLASS} min-w-[220px]`}
        />
        <input
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
          minLength={6}
          placeholder="Password temporária"
          className={`${FIELD_CLASS} min-w-[200px]`}
        />
        <Button type="submit" loading={creating}>
          <UserPlus className="w-4 h-4" />
          Adicionar
        </Button>
      </form>
      {createError && <p className="text-xs text-accent-red mb-2">✗ {createError}</p>}

      <div className="flex items-center gap-2 mt-6 mb-3">
        <SectionLabel>Utilizadores</SectionLabel>
        <span className="font-mono text-[11px] text-fg-subtle">{users.length}</span>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 text-accent animate-spin" />
        </div>
      )}

      {error && <p className="text-sm text-accent-red py-3">{error}</p>}

      {!loading && !error && users.length === 0 && (
        <p className="text-xs text-fg-subtle py-3">Nenhum utilizador encontrado.</p>
      )}

      <div className="flex flex-col gap-2">
        {users.map((user) => (
          <div
            key={user.user_id}
            className="flex items-center gap-3 px-4 py-3 rounded-lg border border-line-soft bg-card"
          >
            <span className="flex-1 min-w-0 font-mono text-[13px] text-fg truncate">
              {user.email}
            </span>
            <span className="hidden sm:block shrink-0 font-mono text-[11px] text-fg-subtle">
              {user.user_id.slice(0, 8)}…
            </span>
            <StatusBadge approved={user.approved} isAdmin={user.is_admin} />
            {!user.is_admin && (
              <Button
                size="sm"
                variant={user.approved ? 'danger' : 'outline'}
                loading={toggling === user.user_id}
                onClick={() => handleToggle(user)}
              >
                {user.approved ? 'Revogar' : 'Aprovar'}
              </Button>
            )}
          </div>
        ))}
      </div>
    </>
  );
}

export function AdminPage() {
  const [tab, setTab] = useState<Tab>('users');

  return (
    <PageBody>
      <div className="flex gap-2 mb-5">
        {TABS.map((entry) => (
          <Pill key={entry.key} active={tab === entry.key} onClick={() => setTab(entry.key)}>
            {entry.label}
          </Pill>
        ))}
      </div>

      <Panel overflow="visible">
        {tab === 'users' ? <UsersTab /> : <TeamAliasPanel />}
      </Panel>
    </PageBody>
  );
}
