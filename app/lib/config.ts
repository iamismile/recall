// Central configuration for the Recall pipeline.
//
// Two sections:
// 1. Pipeline tuning constants (chunking, retrieval depth, limits)
// 2. Environment variables — the ONLY place in the codebase where
//    process.env is read. All other modules import `env` from here.

// ────────────────────────────────────────────────────────────────
// Pipeline constants
// ────────────────────────────────────────────────────────────────

// Number of candidates retrieved by hybrid search before reranking.
export const RERANK_CANDIDATES = 50;

// How many chunks to keep after reranking (the final results).
export const FINAL_TOP_K = 5;

// Per-method fetch cap inside hybrid search (vector + BM25).
// Kept equal to RERANK_CANDIDATES so neither method truncates the
// candidate pool below the rerank window.
export const CANDIDATES = RERANK_CANDIDATES;

// Maximum upload file size in bytes (10 MB).
export const MAX_UPLOAD_SIZE = 10 * 1024 * 1024;

// ────────────────────────────────────────────────────────────────
// Environment variables
// ────────────────────────────────────────────────────────────────

/**
 * Reads a required-at-use-time secret. Returns undefined when unset;
 * call sites decide whether that is fatal (e.g. generation without
 * GEMINI_API_KEY) or optional (e.g. retrieval-only usage).
 */
function optional(key: string): string | undefined {
  const value = process.env[key];
  return value && value.trim() !== "" ? value : undefined;
}

/** Reads a string variable with a default, lowercased for provider names. */
function withDefault(key: string, fallback: string): string {
  return (optional(key) ?? fallback).toLowerCase();
}

/** Reads an integer variable; returns undefined when unset or invalid. */
function int(key: string): number | undefined {
  const value = optional(key);
  if (!value) return undefined;
  const parsed = parseInt(value, 10);
  return Number.isNaN(parsed) ? undefined : parsed;
}

/** Validates a provider name against its allowed values. */
function provider<K extends string>(
  key: string,
  allowed: K[],
  fallback: K,
): K {
  const value = withDefault(key, fallback);
  if (!allowed.includes(value as K)) {
    console.warn(
      `Invalid ${key}="${value}", falling back to "${fallback}" (allowed: ${allowed.join(", ")})`,
    );
    return fallback;
  }
  return value as K;
}

export const env = {
  // Gemini (answer generation + optional embeddings)
  geminiApiKey: optional("GEMINI_API_KEY"),
  geminiModel: optional("GEMINI_MODEL") ?? "gemini-2.5-flash",

  // Embeddings
  embedProvider: provider("EMBED_PROVIDER", ["local", "gemini"], "local"),
  geminiEmbedModel: optional("GEMINI_EMBED_MODEL") ?? "gemini-embedding-001",
  geminiEmbedDimension: int("GEMINI_EMBED_DIMENSION") ?? 768,

  // Reranking
  rerankProvider: provider("RERANK_PROVIDER", ["local", "jina"], "local"),
  jinaRerankingApiKey: optional("JINA_RERANKING_API_KEY"),
  jinaRerankModel:
    optional("JINA_RERANK_MODEL") ?? "jina-reranker-v2-base-multilingual",
} as const;
