import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from "@huggingface/transformers";
import { SearchResult } from "./types";

// Selects which reranking implementation to use
// local or jina
const RERANK_PROVIDER = (process.env.RERANK_PROVIDER ?? "local").toLowerCase();

// Local cross-encoder model used for reranking.
// This model receives both the query and document together
// and produces a relevance score for the pair.
const LOCAL_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type Reranker = Awaited<
  ReturnType<typeof AutoModelForSequenceClassification.from_pretrained>
>;

// Keep the tokenizer + model in memory so we don't reload them per query.
let tokenizer: Tokenizer | null = null;
let reranker: Reranker | null = null;

/**
 * Loads the local reranker model and tokenizer.
 *
 * The model is loaded lazily, meaning it is only loaded when
 * the first reranking request is made.
 *
 * After the first request, the same model is reused.
 */
async function getLocalReranker(): Promise<{
  tokenizer: Tokenizer;
  model: Reranker;
}> {
  if (!tokenizer || !reranker) {
    tokenizer = await AutoTokenizer.from_pretrained(LOCAL_RERANK_MODEL);
    reranker =
      await AutoModelForSequenceClassification.from_pretrained(
        LOCAL_RERANK_MODEL,
      );
  }
  return { tokenizer, model: reranker };
}

/**
 * Minimal shape of the model output that we need.
 *
 * The model produces a logit for each query/document pair.
 * Sigmoid converts the logit into a value between 0 and 1.
 */
interface RerankOutput {
  logits: { sigmoid: () => { data: ArrayLike<number> } };
}

// Result returned by the Jina reranking API.
interface JinaResult {
  index: number;
  relevance_score: number;
}

/**
 * Reranks the candidates returned by our retrieval system.
 *
 * This happens after hybrid retrieval and RRF:
 *
 * Vector Search ──┐
 *                 ├──> RRF ──> Candidates ──> Re-ranking ──> Top N
 * BM25 ───────────┘
 *
 * RRF is good at combining different ranking systems.
 * The reranker then looks more carefully at each query/document
 * pair and produces a more accurate relevance ranking.
 *
 * We support two reranking providers:
 *
 * 1. Local cross-encoder
 *    - Runs entirely on the user's machine.
 *
 * 2. Jina API
 *    - Uses a hosted reranking model.
 */
export async function rerankChunks(
  query: string,
  chunks: SearchResult[],
  topN: number,
): Promise<SearchResult[]> {
  // Nothing to rerank.
  if (chunks.length === 0) return [];

  // Use the configured reranking provider.
  return RERANK_PROVIDER === "jina"
    ? rerankViaJina(query, chunks, topN)
    : rerankLocally(query, chunks, topN);
}

/**
 * Reranks candidates using a local cross-encoder.
 *
 * A normal embedding model is a bi-encoder:
 *
 * Query ──> Vector
 *
 * Document ──> Vector
 *
 * Then we compare the two vectors.
 *
 * A cross-encoder works differently:
 *
 * Query + Document
 *       │
 *       ▼
 * Cross-Encoder
 *       │
 *       ▼
 * Relevance Score
 *
 * Because the model sees the query and document together,
 * it can capture more detailed relationships between them.
 *
 * The trade-off is that it is slower than vector search,
 * which is why we only use it on a small set of candidates
 * returned by RRF.
 */
async function rerankLocally(
  query: string,
  chunks: SearchResult[],
  topN: number,
): Promise<SearchResult[]> {
  const { tokenizer, model } = await getLocalReranker();

  // Create query/document pairs.
  // Every candidate is paired with the same query:
  // Query + Chunk 1
  // Query + Chunk 2
  // ...
  const inputs = tokenizer(
    chunks.map(() => query),
    {
      text_pair: chunks.map((c) => c.text),
      padding: true,
      truncation: true,
    },
  );

  // Run all candidates through the model in one batch.
  const output = (await model(inputs)) as unknown as RerankOutput;

  // Convert the model's logits into values between 0 and 1.
  // Higher scores mean the document is considered more relevant
  // to the query.
  const scores = output.logits.sigmoid().data;

  // Attach the reranking score to each document.
  const scored = chunks.map((chunk, i) => ({
    ...chunk,
    score: scores[i] ?? 0,
  }));

  // Sort by relevance and keep only the requested number of results
  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}

/**
 * Reranks candidates using the Jina Reranker API.
 *
 * Instead of loading a cross-encoder locally, we send the query
 * and candidate documents to Jina's hosted reranking model.
 *
 * This is useful when the application is running in an environment
 * where loading a local ML model would use too much memory.
 *
 * Requires:
 *
 * JINA_RERANKING_API_KEY
 *
 * Optional:
 *
 * JINA_RERANK_MODEL
 */
async function rerankViaJina(
  query: string,
  chunks: SearchResult[],
  topN: number,
): Promise<SearchResult[]> {
  const apiKey = process.env.JINA_RERANKING_API_KEY;
  if (!apiKey) {
    throw new Error("JINA_RERANKING_API_KEY environment variable is not set");
  }

  const model =
    process.env.JINA_RERANK_MODEL ?? "jina-reranker-v2-base-multilingual";

  const res = await fetch("https://api.jina.ai/v1/rerank", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      query,
      documents: chunks.map((c) => c.text),
      top_n: Math.min(topN, chunks.length),
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Jina rerank failed (${res.status}): ${body}`);
  }

  const data = (await res.json()) as { results: JinaResult[] };

  // Jina returns the results ordered from most relevant
  // to least relevant.
  // `index` points back to the original candidate array
  return data.results
    .slice(0, topN)
    .map((r) => ({ ...chunks[r.index], score: r.relevance_score }));
}
