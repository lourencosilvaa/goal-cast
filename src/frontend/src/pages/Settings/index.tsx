import { useState, useEffect } from 'react';
import { Key, Sparkles, Shield, ExternalLink, Cpu, LogOut, Loader2, Zap } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { useAuth } from '@/contexts/AuthContext';
import {
  getGeminiKeyStatus, saveGeminiKey, deleteGeminiKey,
  getNvidiaKeyStatus, saveNvidiaKey, deleteNvidiaKey,
} from '@/lib/api';

const GEMINI_MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite' },
];

export function SettingsPage() {
  const { user, signOut } = useAuth();
  const [geminiKey, setGeminiKey] = useState('');
  const [geminiModel, setGeminiModel] = useState('gemini-2.5-flash');
  const [hasStoredKey, setHasStoredKey] = useState(false);
  const [nvidiaKey, setNvidiaKey] = useState('');
  const [hasNvidiaKey, setHasNvidiaKey] = useState(false);
  const [savingNvidia, setSavingNvidia] = useState(false);
  const [nvidiaFeedback, setNvidiaFeedback] = useState<{ type: 'ok' | 'err'; msg: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'ok' | 'err'; msg: string } | null>(null);

  useEffect(() => {
    const storedModel = localStorage.getItem('gemini_model') || 'gemini-2.5-flash';
    setGeminiModel(storedModel);
    getGeminiKeyStatus().then(setHasStoredKey).catch(() => null);
    getNvidiaKeyStatus().then(setHasNvidiaKey).catch(() => null);
  }, []);

  function showNvidiaFeedback(type: 'ok' | 'err', msg: string) {
    setNvidiaFeedback({ type, msg });
    setTimeout(() => setNvidiaFeedback(null), 3000);
  }

  async function handleSaveNvidia() {
    setSavingNvidia(true);
    try {
      if (nvidiaKey.trim()) {
        await saveNvidiaKey(nvidiaKey.trim());
        setNvidiaKey('');
        setHasNvidiaKey(true);
      }
      showNvidiaFeedback('ok', 'Chave NVIDIA guardada');
    } catch (err) {
      showNvidiaFeedback('err', err instanceof Error ? err.message : 'Erro ao guardar');
    } finally {
      setSavingNvidia(false);
    }
  }

  async function handleClearNvidia() {
    setSavingNvidia(true);
    try {
      await deleteNvidiaKey();
      setNvidiaKey('');
      setHasNvidiaKey(false);
      showNvidiaFeedback('ok', 'Chave removida');
    } catch {
      showNvidiaFeedback('err', 'Erro ao remover chave');
    } finally {
      setSavingNvidia(false);
    }
  }

  function showFeedback(type: 'ok' | 'err', msg: string) {
    setFeedback({ type, msg });
    setTimeout(() => setFeedback(null), 3000);
  }

  async function handleSave() {
    setSaving(true);
    try {
      if (geminiKey.trim()) {
        await saveGeminiKey(geminiKey.trim());
        setGeminiKey('');
        setHasStoredKey(true);
      }
      localStorage.setItem('gemini_model', geminiModel);
      showFeedback('ok', 'Definições guardadas');
    } catch (err) {
      showFeedback('err', err instanceof Error ? err.message : 'Erro ao guardar');
    } finally {
      setSaving(false);
    }
  }

  async function handleClear() {
    setSaving(true);
    try {
      await deleteGeminiKey();
      setGeminiKey('');
      setHasStoredKey(false);
      localStorage.removeItem('gemini_model');
      setGeminiModel('gemini-2.5-flash');
      showFeedback('ok', 'Chave removida');
    } catch {
      showFeedback('err', 'Erro ao remover chave');
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white/90">Definições</h1>
        <p className="text-sm text-white/40 mt-1">
          Configura as API keys para funcionalidades avançadas
        </p>
      </div>

      <div className="max-w-lg space-y-4">
        {/* Account card */}
        <GlassCard gradient="green">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-white/40 mb-0.5">Sessão iniciada como</p>
              <p className="text-sm text-white/80 font-mono">{user?.email}</p>
            </div>
            <NeonButton variant="ghost" onClick={signOut}>
              <LogOut className="w-4 h-4 mr-1.5" />
              Sair
            </NeonButton>
          </div>
        </GlassCard>

        {/* API keys card */}
        <GlassCard gradient="green">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <Key className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white/90">API Keys</h2>
              <p className="text-xs text-white/40">Guardadas de forma encriptada no servidor</p>
            </div>
          </div>

          {/* Gemini key */}
          <div className="mb-5">
            <label className="block text-sm text-white/60 mb-2">
              <Sparkles className="w-3.5 h-3.5 inline mr-1.5" />
              Google Gemini API Key
              {hasStoredKey && (
                <span className="ml-2 text-[10px] text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded-full">
                  Configurada
                </span>
              )}
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/20" />
              <input
                type="password"
                value={geminiKey}
                onChange={(e) => setGeminiKey(e.target.value)}
                placeholder={hasStoredKey ? '••••••••  (deixa em branco para manter)' : 'AIza...'}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl glass border border-white/10 focus:border-green-500/40 focus:outline-none text-sm text-white/80 font-mono placeholder:text-white/20 transition-colors"
              />
            </div>
            <a
              href="https://aistudio.google.com/apikey"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-green-400 hover:text-green-300 mt-2 transition-colors"
            >
              Obter chave <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          {/* Model selector */}
          <div className="mb-5">
            <label className="block text-sm text-white/60 mb-2">
              <Cpu className="w-3.5 h-3.5 inline mr-1.5" />
              Modelo Gemini
            </label>
            <select
              value={geminiModel}
              onChange={(e) => setGeminiModel(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl glass border border-white/10 focus:border-green-500/40 focus:outline-none text-sm text-white/80 bg-transparent transition-colors appearance-none cursor-pointer"
            >
              {GEMINI_MODELS.map((m) => (
                <option key={m.value} value={m.value} className="bg-gray-900 text-white">
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {/* Security note */}
          <GlassCard padding="sm" className="mb-5">
            <div className="flex items-start gap-2">
              <Shield className="w-4 h-4 text-teal-400 mt-0.5 shrink-0" />
              <p className="text-xs text-white/50 leading-relaxed">
                A API key é encriptada com Fernet AES antes de ser guardada no servidor.
                Nunca é exposta ao cliente após ser guardada.
              </p>
            </div>
          </GlassCard>

          {/* Actions */}
          <div className="flex gap-3">
            <NeonButton onClick={handleSave} className="flex-1" disabled={saving}>
              {saving ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : 'Guardar'}
            </NeonButton>
            {hasStoredKey && (
              <NeonButton variant="ghost" onClick={handleClear} disabled={saving}>
                Remover chave
              </NeonButton>
            )}
          </div>

          {feedback && (
            <p className={`text-xs text-center mt-3 ${feedback.type === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
              {feedback.type === 'ok' ? '✓' : '✗'} {feedback.msg}
            </p>
          )}
        </GlassCard>

        {/* NVIDIA API key card */}
        <GlassCard gradient="green">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-600 to-teal-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white/90">NVIDIA API Key</h2>
              <p className="text-xs text-white/40">Para modelos NIM e inferência acelerada por GPU</p>
            </div>
          </div>

          <div className="mb-5">
            <label className="block text-sm text-white/60 mb-2">
              <Zap className="w-3.5 h-3.5 inline mr-1.5" />
              NVIDIA NIM API Key
              {hasNvidiaKey && (
                <span className="ml-2 text-[10px] text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded-full">
                  Configurada
                </span>
              )}
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/20" />
              <input
                type="password"
                value={nvidiaKey}
                onChange={(e) => setNvidiaKey(e.target.value)}
                placeholder={hasNvidiaKey ? '••••••••  (deixa em branco para manter)' : 'nvapi-...'}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl glass border border-white/10 focus:border-green-500/40 focus:outline-none text-sm text-white/80 font-mono placeholder:text-white/20 transition-colors"
              />
            </div>
            <a
              href="https://build.nvidia.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-green-400 hover:text-green-300 mt-2 transition-colors"
            >
              Obter chave em build.nvidia.com <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <GlassCard padding="sm" className="mb-5">
            <div className="flex items-start gap-2">
              <Shield className="w-4 h-4 text-teal-400 mt-0.5 shrink-0" />
              <p className="text-xs text-white/50 leading-relaxed">
                A chave é encriptada com Fernet AES antes de ser guardada. Nunca é exposta ao cliente.
              </p>
            </div>
          </GlassCard>

          <div className="flex gap-3">
            <NeonButton onClick={handleSaveNvidia} className="flex-1" disabled={savingNvidia}>
              {savingNvidia ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : 'Guardar'}
            </NeonButton>
            {hasNvidiaKey && (
              <NeonButton variant="ghost" onClick={handleClearNvidia} disabled={savingNvidia}>
                Remover chave
              </NeonButton>
            )}
          </div>

          {nvidiaFeedback && (
            <p className={`text-xs text-center mt-3 ${nvidiaFeedback.type === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
              {nvidiaFeedback.type === 'ok' ? '✓' : '✗'} {nvidiaFeedback.msg}
            </p>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
