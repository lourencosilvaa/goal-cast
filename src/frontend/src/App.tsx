import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/contexts/AuthContext';
import { PredictionsProvider } from '@/contexts/PredictionsContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { AdminRoute } from '@/components/AdminRoute';
import { AppLayout } from '@/components/layout/AppLayout';
import { Dashboard } from '@/pages/Dashboard';
import { ValueBetsPage } from '@/pages/ValueBets';
import { SettingsPage } from '@/pages/Settings';
import { LoginPage } from '@/pages/Login';
import { AdminPage } from '@/pages/Admin';
import { CustomPredictPage } from '@/pages/CustomPredict';
import { TeamStatsPage } from '@/pages/TeamStats';
import { MatchDetailPage } from '@/pages/MatchDetailPage';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          element={
            <ProtectedRoute>
              <PredictionsProvider>
                <AppLayout />
              </PredictionsProvider>
            </ProtectedRoute>
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="match/:matchId" element={<MatchDetailPage />} />
          <Route path="value-bets" element={<ValueBetsPage />} />
          <Route path="custom-predict" element={<CustomPredictPage />} />
          <Route path="team-stats" element={<TeamStatsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route
            path="admin"
            element={
              <AdminRoute>
                <AdminPage />
              </AdminRoute>
            }
          />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
