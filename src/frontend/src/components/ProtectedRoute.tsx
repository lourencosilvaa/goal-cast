import { Navigate } from 'react-router-dom';
import { Button } from '@/components/ui/Button';
import { Wordmark } from '@/components/layout/Wordmark';
import { useAuth } from '@/contexts/AuthContext';

/** Centred notice card, same proportions as the login card. */
function GateCard({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-bg">
      <div className="w-full max-w-[360px] p-8 rounded-xl border border-line-soft bg-card flex flex-col">
        <Wordmark className="mb-1" />
        <p className="text-[13px] text-fg-subtle">{title}</p>
        <p className="mt-5 text-[13px] text-fg-muted leading-relaxed">{body}</p>
        <div className="mt-5">{action}</div>
      </div>
    </div>
  );
}

function AccessRevoked() {
  const { signOut } = useAuth();
  return (
    <GateCard
      title="Acesso pendente"
      body="A tua conta está a aguardar aprovação do administrador. Serás notificado quando o acesso for concedido."
      action={
        <Button variant="outline" onClick={signOut} className="w-full">
          Terminar sessão
        </Button>
      }
    />
  );
}

function BackendDown() {
  const { refreshProfile } = useAuth();
  return (
    <GateCard
      title="Servidor indisponível"
      body="Não foi possível contactar o servidor. Verifica a tua ligação ou tenta novamente."
      action={
        <Button variant="outline" onClick={refreshProfile} className="w-full">
          Tentar novamente
        </Button>
      }
    />
  );
}

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { session, profile, loading, backendDown } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <div className="w-7 h-7 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }
  if (!session) return <Navigate to="/login" replace />;
  if (backendDown) return <BackendDown />;
  if (!profile || !profile.approved) return <AccessRevoked />;

  return <>{children}</>;
}
