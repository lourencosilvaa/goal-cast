import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export function AdminRoute({ children }: { children: React.ReactNode }) {
  const { profile, loading } = useAuth();

  if (loading) return null;
  if (!profile?.is_admin) return <Navigate to="/" replace />;

  return <>{children}</>;
}
