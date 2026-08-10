import type { AiProvider } from '@/lib/api';

/**
 * Which AI back-end to use, and with which model.
 *
 * Stored per browser rather than per account: it is a preference, not data,
 * and the alternative is a Supabase round trip before every analysis. The
 * model strings live here too, next to the provider they belong to — a model
 * name means nothing without one.
 */

export const AI_PROVIDER_STORAGE_KEY = 'ai_provider';
export const GEMINI_MODEL_STORAGE_KEY = 'gemini_model';
export const NVIDIA_MODEL_STORAGE_KEY = 'nvidia_model';

export const GEMINI_MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite' },
];

/**
 * NVIDIA NIM endpoints, offered as suggestions rather than as the truth.
 *
 * NVIDIA's free preview catalogue (build.nvidia.com) changes without notice,
 * so a fixed list compiled into the bundle goes stale between deploys. The
 * field is therefore editable: pick one of these or paste any model id the
 * account has access to, and the backend passes it straight through.
 */
export const NVIDIA_MODELS = [
  { value: 'nvidia/nemotron-3-nano-30b-a3b', label: 'Nemotron 3 Nano 30B (A3B)' },
  { value: 'meta/llama-3.3-70b-instruct', label: 'Llama 3.3 70B Instruct' },
  { value: 'nvidia/llama-3.3-nemotron-super-49b-v1', label: 'Nemotron Super 49B' },
  { value: 'deepseek-ai/deepseek-r1', label: 'DeepSeek R1' },
  { value: 'mistralai/mixtral-8x22b-instruct-v0.1', label: 'Mixtral 8x22B Instruct' },
  { value: 'qwen/qwen2.5-7b-instruct', label: 'Qwen 2.5 7B Instruct' },
];

export const DEFAULT_AI_PROVIDER: AiProvider = 'gemini';
export const DEFAULT_GEMINI_MODEL = GEMINI_MODELS[0].value;
export const DEFAULT_NVIDIA_MODEL = NVIDIA_MODELS[0].value;

export const AI_PROVIDER_LABELS: Record<AiProvider, string> = {
  gemini: 'Gemini',
  nvidia: 'NVIDIA',
};

function stored(key: string, fallback: string): string {
  if (typeof localStorage === 'undefined') return fallback;
  return localStorage.getItem(key) || fallback;
}

/** The saved default provider, or Gemini when nothing has been chosen. */
export function preferredProvider(): AiProvider {
  const value = stored(AI_PROVIDER_STORAGE_KEY, DEFAULT_AI_PROVIDER);
  return value === 'nvidia' ? 'nvidia' : DEFAULT_AI_PROVIDER;
}

/** The saved model for a provider. Empty is never returned. */
export function preferredModel(provider: AiProvider): string {
  return provider === 'nvidia'
    ? stored(NVIDIA_MODEL_STORAGE_KEY, DEFAULT_NVIDIA_MODEL)
    : stored(GEMINI_MODEL_STORAGE_KEY, DEFAULT_GEMINI_MODEL);
}
