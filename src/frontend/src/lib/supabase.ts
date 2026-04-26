import { createClient } from '@supabase/supabase-js';
import { config } from '@/config';

if (!config.supabaseUrl || !config.supabaseAnonKey) {
  console.error('[supabase] VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY is not set. Check your build environment variables.');
}

export const supabase = createClient(
  config.supabaseUrl || 'https://placeholder.supabase.co',
  config.supabaseAnonKey || 'placeholder',
);
