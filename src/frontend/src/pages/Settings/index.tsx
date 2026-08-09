import { useEffect, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { PageBody } from '@/components/layout/PageBody';
import { FIELD_MONO_CLASS } from '@/components/ui/TeamCombobox';
import { useAuth } from '@/contexts/AuthContext';
import {
  deleteGeminiKey,
  deleteNvidiaKey,
  getGeminiKeyStatus,
  getNvidiaKeyStatus,
  saveGeminiKey,
  saveNvidiaKey,
} from '@/lib/api';

const GEMINI_MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite' },
];

const GEMINI_MODEL_STORAGE_KEY = 'gemini_model';
const DEFAULT_GEMINI_MODEL = GEMINI_MODELS[0].value;
const FEEDBACK_TIMEOUT_MS = 3000;

type Feedback = { type: 'ok' | 'err'; msg: string } | null;

/** Masked key field with a reveal toggle, as in the design's API form. */
function KeyField({
  label,
  value,
  onChange,
  placeholder,
  configured,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  configured: boolean;
}) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-2 text-xs font-semibold text-fg-muted">
        {label}
        {configured && (
          <span className="px-1.5 py-0.5 rounded border border-accent-green/40 font-mono text-[9px] font-bold text-accent-green">
            CONFIGURADA
          </span>
        )}
      </label>
      <div className="flex gap-2">
        <input
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={configured ? '••••••••  (deixa em branco para manter)' : placeholder}
          className={`${FIELD_MONO_CLASS} flex-1 min-w-0`}
        />
        <Button variant="outline" size="md" type="button" onClick={() => setVisible((v) => !v)}>
          {visible ? 'Ocultar' : 'Mostrar'}
        </Button>
      </div>
    </div>
  );
}

export function SettingsPage() {
  const { user, signOut } = useAuth();

  const [geminiKey, setGeminiKey] = useState('');
  const [geminiModel, setGeminiModel] = useState(DEFAULT_GEMINI_MODEL);
  const [hasGeminiKey, setHasGeminiKey] = useState(false);
  const [savingGemini, setSavingGemini] = useState(false);
  const [geminiFeedback, setGeminiFeedback] = useState<Feedback>(null);

  const [nvidiaKey, setNvidiaKey] = useState('');
  const [hasNvidiaKey, setHasNvidiaKey] = useState(false);
  const [savingNvidia, setSavingNvidia] = useState(false);
  const [nvidiaFeedback, setNvidiaFeedback] = useState<Feedback>(null);

  useEffect(() => {
    setGeminiModel(localStorage.getItem(GEMINI_MODEL_STORAGE_KEY) || DEFAULT_GEMINI_MODEL);
    getGeminiKeyStatus().then(setHasGeminiKey).catch(() => null);
    getNvidiaKeyStatus().then(setHasNvidiaKey).catch(() => null);
  }, []);

  function flash(set: (value: Feedback) => void, type: 'ok' | 'err', msg: string) {
    set({ type, msg });
    setTimeout(() => set(null), FEEDBACK_TIMEOUT_MS);
  }

  async function handleSaveGemini() {
    setSavingGemini(true);
    try {
      if (geminiKey.trim()) {
        await saveGeminiKey(geminiKey.trim());
        setGeminiKey('');
        setHasGeminiKey(true);
      }
      localStorage.setItem(GEMINI_MODEL_STORAGE_KEY, geminiModel);
      flash(setGeminiFeedback, 'ok', 'Definições guardadas');
    } catch (err) {
      flash(
        setGeminiFeedback,
        'err',
        err instanceof Error ? err.message : 'Erro ao guardar',
      );
    } finally {
      setSavingGemini(false);
    }
  }

  async function handleClearGemini() {
    setSavingGemini(true);
    try {
      await deleteGeminiKey();
      setGeminiKey('');
      setHasGeminiKey(false);
      localStorage.removeItem(GEMINI_MODEL_STORAGE_KEY);
      setGeminiModel(DEFAULT_GEMINI_MODEL);
      flash(setGeminiFeedback, 'ok', 'Chave removida');
    } catch {
      flash(setGeminiFeedback, 'err', 'Erro ao remover chave');
    } finally {
      setSavingGemini(false);
    }
  }

  async function handleSaveNvidia() {
    setSavingNvidia(true);
    try {
      if (nvidiaKey.trim()) {
        await saveNvidiaKey(nvidiaKey.trim());
        setNvidiaKey('');
        setHasNvidiaKey(true);
      }
      flash(setNvidiaFeedback, 'ok', 'Chave NVIDIA guardada');
    } catch (err) {
      flash(setNvidiaFeedback, 'err', err instanceof Error ? err.message : 'Erro ao guardar');
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
      flash(setNvidiaFeedback, 'ok', 'Chave removida');
    } catch {
      flash(setNvidiaFeedback, 'err', 'Erro ao remover chave');
    } finally {
      setSavingNvidia(false);
    }
  }

  return (
    <PageBody
      maxWidth="md"
      intro="Chaves de API para as funcionalidades avançadas. São encriptadas com Fernet AES antes de serem guardadas no servidor e nunca voltam a ser expostas ao cliente."
    >
      <Panel className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <SectionLabel>Sessão iniciada como</SectionLabel>
          <p className="mt-1 font-mono text-[13px] text-fg truncate">{user?.email}</p>
        </div>
        <Button variant="outline" size="sm" onClick={signOut}>
          Terminar sessão
        </Button>
      </Panel>

      <Panel className="mt-4 flex flex-col gap-4">
        <SectionLabel>Google Gemini</SectionLabel>

        <KeyField
          label="API key"
          value={geminiKey}
          onChange={setGeminiKey}
          placeholder="AIza…"
          configured={hasGeminiKey}
        />

        <a
          href="https://aistudio.google.com/apikey"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 w-fit text-xs text-accent hover:opacity-80"
        >
          Obter chave <ExternalLink className="w-3 h-3" />
        </a>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-fg-muted">Modelo</label>
          <select
            value={geminiModel}
            onChange={(e) => setGeminiModel(e.target.value)}
            className="w-full px-3 py-2.5 rounded-md bg-card border border-line text-fg text-sm outline-none focus:border-accent/45 transition-colors cursor-pointer"
          >
            {GEMINI_MODELS.map((model) => (
              <option key={model.value} value={model.value} className="bg-card text-fg">
                {model.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-2">
          <Button onClick={handleSaveGemini} loading={savingGemini} className="flex-1">
            Guardar
          </Button>
          {hasGeminiKey && (
            <Button variant="danger" onClick={handleClearGemini} disabled={savingGemini}>
              Remover chave
            </Button>
          )}
        </div>

        {geminiFeedback && (
          <p
            className={`text-xs ${geminiFeedback.type === 'ok' ? 'text-accent-green' : 'text-accent-red'}`}
          >
            {geminiFeedback.type === 'ok' ? '✓' : '✗'} {geminiFeedback.msg}
          </p>
        )}
      </Panel>

      <Panel className="mt-4 flex flex-col gap-4">
        <SectionLabel>NVIDIA NIM</SectionLabel>

        <KeyField
          label="API key"
          value={nvidiaKey}
          onChange={setNvidiaKey}
          placeholder="nvapi-…"
          configured={hasNvidiaKey}
        />

        <a
          href="https://build.nvidia.com"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 w-fit text-xs text-accent hover:opacity-80"
        >
          Obter chave em build.nvidia.com <ExternalLink className="w-3 h-3" />
        </a>

        <div className="flex gap-2">
          <Button onClick={handleSaveNvidia} loading={savingNvidia} className="flex-1">
            Guardar
          </Button>
          {hasNvidiaKey && (
            <Button variant="danger" onClick={handleClearNvidia} disabled={savingNvidia}>
              Remover chave
            </Button>
          )}
        </div>

        {nvidiaFeedback && (
          <p
            className={`text-xs ${nvidiaFeedback.type === 'ok' ? 'text-accent-green' : 'text-accent-red'}`}
          >
            {nvidiaFeedback.type === 'ok' ? '✓' : '✗'} {nvidiaFeedback.msg}
          </p>
        )}
      </Panel>
    </PageBody>
  );
}
