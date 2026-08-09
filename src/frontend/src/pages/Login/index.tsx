import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/Button';
import { FIELD_CLASS } from '@/components/ui/TeamCombobox';
import { Wordmark } from '@/components/layout/Wordmark';
import { useAuth } from '@/contexts/AuthContext';
import { registerUser } from '@/lib/api';

type Mode = 'login' | 'register';

function AuthCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-bg">
      <div className="w-full max-w-[360px] p-8 rounded-xl border border-line-soft bg-card flex flex-col">
        {children}
      </div>
    </div>
  );
}

export function LoginPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === 'login') {
        await signIn(email, password);
        navigate('/');
      } else {
        await registerUser(email, password);
        setRegistered(true);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : mode === 'login'
            ? 'Falha ao iniciar sessão'
            : 'Falha no registo',
      );
    } finally {
      setLoading(false);
    }
  }

  function switchMode(next: Mode) {
    setMode(next);
    setError(null);
    setRegistered(false);
    setEmail('');
    setPassword('');
  }

  if (registered) {
    return (
      <AuthCard>
        <Wordmark className="mb-1" />
        <p className="text-[13px] text-fg-subtle">Registo efetuado</p>
        <p className="mt-5 text-[13px] text-fg-muted leading-relaxed">
          A tua conta foi criada e aguarda aprovação do administrador. Receberás acesso em breve.
        </p>
        <Button variant="outline" onClick={() => switchMode('login')} className="mt-5 w-full">
          Voltar ao início de sessão
        </Button>
      </AuthCard>
    );
  }

  return (
    <AuthCard>
      <Wordmark className="mb-1" />
      <p className="text-[13px] text-fg-subtle">
        {mode === 'login' ? 'Inicia sessão para ver as previsões' : 'Cria a tua conta'}
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 mt-5">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder="Email"
          autoComplete="email"
          className={`${FIELD_CLASS} w-full`}
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={6}
          placeholder="Password"
          autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          className={`${FIELD_CLASS} w-full`}
        />

        {error && <p className="text-xs text-accent-red">{error}</p>}

        <Button type="submit" loading={loading} className="mt-2 w-full py-3">
          {mode === 'login' ? 'Entrar' : 'Criar conta'}
        </Button>
      </form>

      <p className="mt-4 text-[11px] text-fg-subtle leading-relaxed">
        {mode === 'login' ? (
          <>
            As contas novas são revistas por um administrador antes de o acesso ser concedido.{' '}
            <button
              type="button"
              onClick={() => switchMode('register')}
              className="text-accent hover:opacity-80 cursor-pointer"
            >
              Regista-te
            </button>
            .
          </>
        ) : (
          <>
            Já tens conta?{' '}
            <button
              type="button"
              onClick={() => switchMode('login')}
              className="text-accent hover:opacity-80 cursor-pointer"
            >
              Inicia sessão
            </button>
            .
          </>
        )}
      </p>
    </AuthCard>
  );
}
