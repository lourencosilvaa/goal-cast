import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, UserPlus, Mail, Lock, Loader2, CheckCircle2 } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { useAuth } from '@/contexts/AuthContext';
import { registerUser } from '@/lib/api';

type Mode = 'login' | 'register';

export function LoginPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signIn(email, password);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await registerUser(email, password);
      setRegistered(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
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
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="w-full max-w-sm text-center">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent-green to-accent-green flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="w-7 h-7 text-white" />
          </div>
          <GlassCard accent="green">
            <h2 className="text-lg font-semibold text-fg mb-2">Registo efetuado!</h2>
            <p className="text-sm text-fg-muted mb-5 leading-relaxed">
              A tua conta foi criada e aguarda aprovação do administrador.
              Receberás acesso em breve.
            </p>
            <NeonButton variant="ghost" onClick={() => switchMode('login')} className="w-full">
              Voltar ao login
            </NeonButton>
          </GlassCard>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center mx-auto mb-4">
            {mode === 'login' ? (
              <LogIn className="w-7 h-7 text-white" />
            ) : (
              <UserPlus className="w-7 h-7 text-white" />
            )}
          </div>
          <h1 className="text-2xl font-bold text-fg">Football Agent</h1>
          <p className="text-sm text-fg-muted mt-1">
            {mode === 'login' ? 'Inicia sessão para continuar' : 'Cria a tua conta'}
          </p>
        </div>

        <GlassCard accent="blue">
          <form onSubmit={mode === 'login' ? handleLogin : handleRegister} className="space-y-4">
            <div>
              <label className="block text-sm text-fg-muted mb-1.5">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-subtle" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="email@exemplo.com"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl glass border border-line focus:border-accent-blue/40 focus:outline-none text-sm text-fg placeholder:text-fg-subtle transition-colors"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm text-fg-muted mb-1.5">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-subtle" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl glass border border-line focus:border-accent-blue/40 focus:outline-none text-sm text-fg placeholder:text-fg-subtle transition-colors"
                />
              </div>
            </div>

            {error && (
              <p className="text-xs text-accent-red text-center">{error}</p>
            )}

            <NeonButton type="submit" className="w-full" disabled={loading}>
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin mx-auto" />
              ) : mode === 'login' ? (
                'Entrar'
              ) : (
                'Criar conta'
              )}
            </NeonButton>
          </form>

          <div className="mt-4 pt-4 border-t border-line text-center">
            {mode === 'login' ? (
              <p className="text-xs text-fg-muted">
                Ainda não tens conta?{' '}
                <button
                  onClick={() => switchMode('register')}
                  className="text-accent-blue hover:opacity-80 transition-opacity"
                >
                  Regista-te
                </button>
              </p>
            ) : (
              <p className="text-xs text-fg-muted">
                Já tens conta?{' '}
                <button
                  onClick={() => switchMode('login')}
                  className="text-accent-blue hover:opacity-80 transition-opacity"
                >
                  Inicia sessão
                </button>
              </p>
            )}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
