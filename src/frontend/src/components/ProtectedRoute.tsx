import { Navigate } from 'react-router-dom';
import { ShieldOff, ServerCrash, LogOut, RefreshCw } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';

function AccessRevoked() {
  const { signOut } = useAuth();
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm text-center">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center mx-auto mb-4">
          <ShieldOff className="w-7 h-7 text-white" />
        </div>
        <GlassCard gradient="green">
          <h2 className="text-lg font-semibold text-white/90 mb-2">Acesso Pendente</h2>
          <p className="text-sm text-white/50 mb-5 leading-relaxed">
            A tua conta está a aguardar aprovação do administrador.
            Serás notificado quando o acesso for concedido.
          </p>
          <NeonButton variant="ghost" onClick={signOut} className="w-full">
            <LogOut className="w-4 h-4 mr-1.5" />
            Sair
          </NeonButton>
        </GlassCard>
      </div>
    </div>
  );
}

function BackendDown() {
  const { refreshProfile } = useAuth();
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm text-center">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-yellow-500 to-orange-600 flex items-center justify-center mx-auto mb-4">
          <ServerCrash className="w-7 h-7 text-white" />
        </div>
        <GlassCard gradient="green">
          <h2 className="text-lg font-semibold text-white/90 mb-2">Servidor Indisponível</h2>
          <p className="text-sm text-white/50 mb-5 leading-relaxed">
            Não foi possível contactar o servidor. Verifica a tua ligação ou tenta novamente.
          </p>
          <NeonButton variant="ghost" onClick={refreshProfile} className="w-full">
            <RefreshCw className="w-4 h-4 mr-1.5" />
            Tentar Novamente
          </NeonButton>
        </GlassCard>
      </div>
    </div>
  );
}

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { session, profile, loading, backendDown } = useAuth();

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
  if (!session) return <Navigate to="/login" replace />;
  if (backendDown) return <BackendDown />;
  if (!profile || !profile.approved) return <AccessRevoked />;

  return <>{children}</>;
}
