// Number of candidates retrieved by hybrid search before reranking.
export const RERANK_CANDIDATES = 50;

// How many chunks to keep after reranking (the final results).
export const FINAL_TOP_K = 5;

// Per-method fetch cap inside hybrid search (vector + BM25).
// Kept equal to RERANK_CANDIDATES so neither method truncates the
// candidate pool below the rerank window.
export const CANDIDATES = RERANK_CANDIDATES;
