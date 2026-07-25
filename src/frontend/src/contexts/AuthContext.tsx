import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { Session, User } from '@supabase/supabase-js';
import { config, devAuth } from '@/config';
import { supabase } from '@/lib/supabase';

interface UserProfile {
  user_id: string;
  email: string;
  approved: boolean;
  is_admin: boolean;
}

interface AuthContextValue {
  session: Session | null;
  user: User | null;
  profile: UserProfile | null;
  loading: boolean;
  backendDown: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** Fake session + profile used only when the local dev bypass is enabled. */
const DEV_SESSION = {
  access_token: devAuth.token,
  token_type: 'bearer',
  user: { id: devAuth.user.user_id, email: devAuth.user.email },
} as unknown as Session;

const DEV_PROFILE: UserProfile = {
  user_id: devAuth.user.user_id,
  email: devAuth.user.email,
  approved: true,
  is_admin: true,
};

async function fetchProfile(token: string): Promise<{ profile: UserProfile | null; backendDown: boolean }> {
  try {
    const res = await fetch(`${config.apiUrl}/api/users/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status >= 500) return { profile: null, backendDown: true };
    if (!res.ok) return { profile: null, backendDown: false };
    return { profile: await res.json(), backendDown: false };
  } catch {
    return { profile: null, backendDown: true };
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  // In local dev-bypass mode the mock session/profile are seeded up front so the
  // effect can no-op (avoids a synchronous setState inside the effect).
  const [session, setSession] = useState<Session | null>(devAuth.enabled ? DEV_SESSION : null);
  const [profile, setProfile] = useState<UserProfile | null>(devAuth.enabled ? DEV_PROFILE : null);
  const [loading, setLoading] = useState(!devAuth.enabled);
  const [backendDown, setBackendDown] = useState(false);

  async function loadProfile(s: Session | null) {
    if (!s) { setProfile(null); setBackendDown(false); return; }
    const { profile: p, backendDown: down } = await fetchProfile(s.access_token);
    setProfile(p);
    setBackendDown(down);
  }

  useEffect(() => {
    // Local-only: skip Supabase + backend profile fetch entirely.
    if (devAuth.enabled) return;

    supabase.auth.getSession().then(async ({ data }) => {
      setSession(data.session);
      await loadProfile(data.session);
      setLoading(false);
    });

    const { data: listener } = supabase.auth.onAuthStateChange(async (_event, s) => {
      setSession(s);
      await loadProfile(s);
    });

    return () => listener.subscription.unsubscribe();
  }, []);

  async function signIn(email: string, password: string) {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
  }

  async function signOut() {
    if (devAuth.enabled) return;
    await supabase.auth.signOut();
    setProfile(null);
  }

  async function refreshProfile() {
    if (session) await loadProfile(session);
  }

  return (
    <AuthContext.Provider
      value={{
        session,
        user: session?.user ?? null,
        profile,
        loading,
        backendDown,
        signIn,
        signOut,
        refreshProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
