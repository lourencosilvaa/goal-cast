import { useState, useEffect } from 'react';
import { Key, Sparkles, CheckCircle, ExternalLink, Cpu } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { NeonButton } from '@/components/ui/NeonButton';
import { AuroraBackground } from '@/components/ui/AuroraBackground';

const GEMINI_MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite' },
];

export function SettingsPage() {
  const [geminiKey, setGeminiKey] = useState('');
  const [geminiModel, setGeminiModel] = useState('gemini-2.5-flash');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('gemini_api_key') || '';
    setGeminiKey(stored);
    const storedModel = localStorage.getItem('gemini_model') || 'gemini-2.5-flash';
    setGeminiModel(storedModel);
  }, []);

  function handleSave() {
    if (geminiKey.trim()) {
      localStorage.setItem('gemini_api_key', geminiKey.trim());
    } else {
      localStorage.removeItem('gemini_api_key');
    }
    localStorage.setItem('gemini_model', geminiModel);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  function handleClear() {
    setGeminiKey('');
    setGeminiModel('gemini-2.5-flash');
    localStorage.removeItem('gemini_api_key');
    localStorage.removeItem('gemini_model');
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white/90">Definições</h1>
        <p className="text-sm text-white/40 mt-1">
          Configura as API keys para funcionalidades avançadas
        </p>
      </div>

      <div className="max-w-lg">
        <GlassCard gradient="green">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <Key className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white/90">
                API Keys
              </h2>
              <p className="text-xs text-white/40">
                Guardadas localmente no browser
              </p>
            </div>
          </div>

          {/* Gemini key */}
          <div className="mb-5">
            <label className="block text-sm text-white/60 mb-2">
              <Sparkles className="w-3.5 h-3.5 inline mr-1.5" />
              Google Gemini API Key
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/20" />
              <input
                type="password"
                value={geminiKey}
                onChange={(e) => setGeminiKey(e.target.value)}
                placeholder="AIza..."
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

          {/* Info */}
          <GlassCard padding="sm" className="mb-5">
            <div className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-teal-400 mt-0.5 shrink-0" />
              <p className="text-xs text-white/50 leading-relaxed">
                A API key do Gemini permite gerar análises personalizadas para
                cada jogo com IA. A chave é guardada apenas no teu browser
                (localStorage) e enviada diretamente à API do Google — nunca é
                armazenada no nosso servidor.
              </p>
            </div>
          </GlassCard>

          {/* Future: HuggingFace */}
          <div className="mb-5 opacity-40">
            <label className="block text-sm text-white/60 mb-2">
              🤗 HuggingFace API Key
              <span className="text-[10px] ml-2 text-white/30">(em breve)</span>
            </label>
            <input
              type="password"
              disabled
              placeholder="Em breve..."
              className="w-full px-4 py-2.5 rounded-xl glass border border-white/10 text-sm text-white/30 cursor-not-allowed"
            />
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <NeonButton onClick={handleSave} className="flex-1">
              Guardar
            </NeonButton>
            <NeonButton variant="ghost" onClick={handleClear}>
              Limpar
            </NeonButton>
          </div>

          {saved && (
            <p className="text-xs text-emerald-400 text-center mt-3">
              ✓ Definições guardadas
            </p>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
