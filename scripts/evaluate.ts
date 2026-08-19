/**
 * Retrieval evaluation for Recall.
 *
 * Measures how effectively the retrieval pipeline finds relevant
 * document chunks for a user's question.
 *
 * Retrieval pipeline:
 *
 *   Vector Search ─┐
 *                  ├─→ RRF ─→ Top 20 candidates ─→ Cross-Encoder ─→ Top 5
 *   BM25 ──────────┘
 *
 * We measure the results at two stages:
 *
 * 1. RRF
 *    The ranking produced by hybrid retrieval.
 *
 * 2. Reranked
 *    The ranking after the cross-encoder reorders the RRF candidates.
 *
 * Comparing both stages tells us whether the reranker actually
 * improves retrieval quality.
 *
 * This evaluation does not call an LLM. It only evaluates retrieval,
 * so the benchmark can run offline once the embedding and reranking
 * models are available locally.
 *
 * Usage:
 *
 *   npm run eval
 *     Ingest sample documents and run the evaluation.
 *
 *   npm run eval -- --no-ingest
 *     Evaluate the existing indexes without re-ingesting documents.
 *
 *   npm run eval -- --fresh
 *     Delete existing indexes, ingest the sample documents, and
 *     run a clean evaluation.
 *
 * Metrics:
 *
 *   Recall@K
 *     Whether at least one relevant chunk appears within the top K.
 *
 *   MRR
 *     How high the first relevant chunk appears in the ranking.
 *
 *   nDCG@K
 *     How well relevant chunks are ordered within the top K.
 *
 * Ground truth:
 *
 * A chunk is considered relevant when its text contains the expected
 * answer snippet defined in eval/qa.jsonl.
 *
 * This makes the benchmark deterministic, but it is an exact text-match
 * evaluation rather than a semantic relevance judgment.
 */

import "dotenv/config";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

import { chunkText } from "@/app/lib/chunker";
import { embedTexts } from "@/app/lib/embeddings";
import { addChunksToIndex, deleteIndexByDocId } from "@/app/lib/minisearch";
import { rerankChunks } from "@/app/lib/rerank";
import { searchHybrid } from "@/app/lib/search";
import { Chunk } from "@/app/lib/types";
import {
  addChunks,
  deleteByDocId as deleteVectorByDocId,
  getSources as getVectorSources,
} from "@/app/lib/vectordb";

const RERANK_CANDIDATES = 20;
const FINAL_TOP_K = 5;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EVAL_DIR = path.join(__dirname, "..", "eval");
const SAMPLE_DIR = path.join(EVAL_DIR, "sample-docs");
const DATASET_PATH = path.join(EVAL_DIR, "qa.jsonl");

interface QaItem {
  query: string;
  expectedSnippet: string;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/**
 * Converts retrieval results into binary relevance labels.
 *
 * Example:
 *
 *   Results:       [A, B, C, D]
 *   Relevant:       A     C
 *
 *   Relevance:     [1, 0, 1, 0]
 *
 * 1 = relevant
 * 0 = irrelevant
 */
function relevanceArray(texts: string[], expectedSnippet: string): number[] {
  const needle = expectedSnippet.toLowerCase().replace(/\s+/g, " ").trim();

  return texts.map((text) => {
    const hay = text.toLowerCase().replace(/\s+/g, " ").trim();
    return hay.includes(needle) ? 1 : 0;
  });
}

/**
 * Calculates Discounted Cumulative Gain (DCG).
 *
 * DCG measures how good the order of our search results is.
 *
 * The important idea:
 *
 *   A relevant result near the top is more valuable
 *   than the same relevant result near the bottom.
 *
 * Example:
 *
 *   [1, 1, 0, 0]  ← better
 *   [0, 0, 1, 1]  ← worse
 *
 * Both rankings contain the same number of relevant results,
 * but the first ranking puts them closer to the top.
 *
 * We give each relevant result a smaller score as its position
 * gets lower in the ranking.
 *
 * For binary relevance:
 *
 *   1 = relevant
 *   0 = not relevant
 *
 * The formula is:
 *
 *   DCG = Σ relevance / log2(rank + 1)
 */
function dcg(rels: number[]): number {
  return rels.reduce(
    (acc, rel, i) => acc + (rel > 0 ? rel / Math.log2(i + 2) : 0),
    0,
  );
}

/**
 * Recall@K for a single query.
 *
 * Returns 1 when at least one relevant chunk appears in the
 * first K results; otherwise returns 0.
 *
 * Example:
 *
 * Results:
 *   [0, 0, 1, 0, 0]
 * Recall@3 = 1, because a relevant result appears within the first 3
 *
 * Results:
 *   [0, 0, 0, 1, 0]
 * Recall@3 = 0, because the relevant result is outside the top 3.
 *
 * This is particularly important for RAG because the answer
 * generator can only use information that retrieval provides.
 */
function recallAtK(rels: number[], k: number): number {
  return rels.slice(0, k).some((r) => r > 0) ? 1 : 0;
}

/**
 * Reciprocal Rank for a single query.
 *
 * Measures the position of the first relevant result:
 *
 *   rank 1 → 1.0
 *   rank 2 → 0.5
 *   rank 3 → 0.333
 *
 * No relevant result → 0
 *
 * MRR is the mean reciprocal rank across all queries.
 */
function mrr(rels: number[]): number {
  const first = rels.findIndex((r) => r > 0);
  return first === -1 ? 0 : 1 / (first + 1);
}

/**
 * Calculates normalized Discounted Cumulative Gain (nDCG) at K.
 *
 * First, remember what DCG does:
 *
 *   DCG tells us how good the order of our search results is.
 *   Relevant results near the top are better than relevant
 *   results near the bottom.
 *
 * But DCG has one problem:
 *
 *   A DCG score by itself doesn't tell us how good the score is.
 *
 * nDCG solves this by comparing our ranking with the
 * "perfect" ranking.
 *
 * Example:
 *
 *   Our ranking:
 *   [1, 0, 1, 0]
 *
 *   Perfect ranking:
 *   [1, 1, 0, 0]
 *
 * The perfect ranking puts all relevant results first.
 *
 * nDCG is:
 *
 *   nDCG = our DCG / perfect DCG
 *
 * This gives us a score between 0 and 1:
 *
 *   1.0 → our ranking is perfect
 *   0.0 → no relevant result was found
 *
 * We calculate this only for the first K results.
 *
 * The benchmark uses binary relevance:
 *
 *   1 = relevant
 *   0 = not relevant
 */
function ndcgAtK(rels: number[], k: number): number {
  // Count how many relevant results we have.
  const totalRelevant = rels.filter((r) => r > 0).length;

  // If there are no relevant results, there is nothing to measure.
  if (totalRelevant === 0) return 0;

  // Build the perfect ranking.
  //
  // If we have 2 relevant results and K = 5:
  //   [1, 1]
  //
  // This represents the best possible ranking for our
  // available relevant results.
  const idealCount = Math.min(totalRelevant, k);
  const ideal = Array.from({ length: idealCount }, () => 1);

  // DCG of the perfect ranking.
  const idcg = dcg(ideal);

  // Compare our ranking with the perfect ranking.
  return idcg === 0 ? 0 : dcg(rels.slice(0, k)) / idcg;
}

// ---------------------------------------------------------------------------
// Ingestion of the sample document
// ---------------------------------------------------------------------------

/**
 * Ingests the benchmark documents using the same processing pipeline
 * used by normal document uploads.
 *
 *   Markdown
 *      ↓
 *   Chunking
 *      ↓
 *   Embeddings
 *      ↓
 *   LanceDB + MiniSearch
 *
 * The previous version of each document is removed before indexing
 * so repeated evaluations do not create duplicate chunks.
 */
async function ingestSampleDocs(evalDocIds: Set<string>): Promise<void> {
  // Find all Markdown documents in the evaluation dataset.
  const files = (await fs.readdir(SAMPLE_DIR)).filter((f) => f.endsWith(".md"));

  for (const file of files) {
    // Remove the previous document from both retrieval indexes.
    await deleteVectorByDocId(file);
    await deleteIndexByDocId(file);

    // Split the document using the application's normal chunking strategy.
    const text = await fs.readFile(path.join(SAMPLE_DIR, file), "utf-8");
    const chunks = chunkText(text);
    const chunkObjects: Chunk[] = chunks.map((chunkText, i) => ({
      id: `${file}-${i}-${Math.random().toString(36).slice(2, 8)}`,
      docId: file,
      text: chunkText,
      source: file,
      chunkIndex: i,
    }));

    // Keep the same chunk objects in both retrieval systems so
    // vector search and BM25 operate over identical content.
    const vectors = await embedTexts(chunks);
    await addChunks(chunkObjects, vectors);
    await addChunksToIndex(chunkObjects);
    console.log(`• indexed ${file} (${chunks.length} chunks)`);

    // Keep track of the document being evaluated.
    // This allows the evaluation to measure only the benchmark
    // documents and ignore unrelated documents in the indexes.
    evalDocIds.add(file);
  }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/**
 * Loads the evaluation questions from the JSONL dataset.
 *
 * Each line contains one question and the expected relevant
 * text snippet. JSONL is convenient here because every question
 * can be stored as an independent JSON object.
 */
async function loadDataset(): Promise<QaItem[]> {
  const raw = await fs.readFile(DATASET_PATH, "utf-8");
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as QaItem);
}

/**
 * Evaluates the retrieval pipeline.
 *
 * Each question is tested twice:
 *
 * 1. RRF results
 *    - Vector Search + BM25
 *    - Combined using Reciprocal Rank Fusion
 *
 * 2. Reranked results
 *    - RRF candidates are passed to the cross-encoder reranker
 *    - The reranker produces the final ranking
 *
 * Comparing these two stages tells us whether reranking
 * actually improves retrieval quality.
 *
 * Metrics:
 *
 * Recall@K
 * - Checks whether a relevant chunk appears within the top K results.
 *
 * MRR (Mean Reciprocal Rank)
 * - Measures how high the first relevant result appears.
 * - A relevant result at rank 1 gets 1.0.
 * - Rank 2 gets 0.5, rank 3 gets 0.333, etc.
 *
 * nDCG@K
 * - Measures how well relevant results are ordered.
 * - Higher scores mean relevant chunks appear closer to the top.
 */
async function evaluate(evalDocIds: Set<string>): Promise<void> {
  // Load the questions and expected answers from the evaluation dataset.
  const dataset = await loadDataset();

  // There is nothing to evaluate if the dataset is empty.
  if (dataset.length === 0) {
    console.error("Dataset is empty.");
    process.exit(1);
  }

  // Evaluate retrieval at several cutoff points.
  const ks = [1, 3, 5];

  // Store metrics separately for:
  //   - RRF results before reranking
  //   - Final results after reranking
  // We can compare these values later to see whether
  // the reranker improves the retrieval pipeline.
  const agg = {
    rrf: {
      recall: Object.fromEntries(ks.map((k) => [k, 0])),
      mrr: 0,
      ndcg: Object.fromEntries(ks.map((k) => [k, 0])),
    },
    rerank: {
      recall: Object.fromEntries(ks.map((k) => [k, 0])),
      mrr: 0,
      ndcg: Object.fromEntries(ks.map((k) => [k, 0])),
    },
  };

  console.log(
    "\nPer-query results (✓ = relevant chunk in top 5 after rerank)\n",
  );

  // Evaluate every question in the dataset independently.
  for (const item of dataset) {
    // Generate the query embedding once.
    const queryVector = (await embedTexts([item.query]))[0];

    // Stage 1: Hybrid retrieval.
    // Vector Search + BM25 are combined using Reciprocal Rank Fusion.
    //
    // Restrict evaluation to benchmark documents.
    // This prevents unrelated documents already present in the
    // indexes from affecting the evaluation results.
    const candidates = (
      await searchHybrid(item.query, queryVector, RERANK_CANDIDATES)
    ).filter((c) => evalDocIds.has(c.docId));

    // Evaluate the original RRF ranking.
    const rrfRels = relevanceArray(
      candidates.map((c) => c.text),
      item.expectedSnippet,
    );

    // Stage 2: Cross-encoder reranking.
    // The reranker receives the top RRF candidates and produces
    // the final top-K ranking.
    const reranked = await rerankChunks(item.query, candidates, FINAL_TOP_K);

    // Evaluate the reranked results.
    const rerankRels = relevanceArray(
      reranked.map((c) => c.text),
      item.expectedSnippet,
    );

    // Compare the actual chunk ordering before and after reranking.
    const rrfTopK = rrfRels.slice(0, FINAL_TOP_K);
    const rankingChanged =
      JSON.stringify(rrfTopK) !== JSON.stringify(rerankRels);

    // Accumulate Recall@K and nDCG@K for both retrieval stages.
    for (const k of ks) {
      agg.rrf.recall[k] += recallAtK(rrfRels, k);
      agg.rerank.recall[k] += recallAtK(rerankRels, k);

      agg.rrf.ndcg[k] += ndcgAtK(rrfRels, k);
      agg.rerank.ndcg[k] += ndcgAtK(rerankRels, k);
    }

    // Accumulate MRR to measure whether reranking moves the
    // first relevant chunk closer to the top.
    agg.rrf.mrr += mrr(rrfRels);
    agg.rerank.mrr += mrr(rerankRels);

    // Print a simple per-question result.
    // ✓ means the final reranked top-5 contains a relevant chunk.
    // ✗ means it does not.
    const hit = rerankRels.slice(0, FINAL_TOP_K).some((r) => r > 0);
    console.log(`  ${hit ? "✓" : "✗"}  ${item.query}`);

    // Check whether the cross-encoder changed the ordering
    // of the top-K candidate chunks.
    if (rankingChanged) {
      console.log(`    ⚠ rerank changed ranking`);
      const rrfLabel = `RRF (top ${FINAL_TOP_K})`.padEnd(16);
      const rerankLabel = "Reranked".padEnd(16);
      console.log(`      ${rrfLabel}  ${rrfTopK}`);
      console.log(`      ${rerankLabel}  ${rerankRels}`);
    }
  }

  /**
   * Convert the accumulated metric totals into averages.
   *
   * For example:
   *
   * Recall@5 =
   *   number of questions that retrieved a relevant chunk in top 5
   *   -----------------------------------------------------------
   *                     total number of questions
   */
  const n = dataset.length;
  for (const k of ks) {
    agg.rrf.recall[k] /= n;
    agg.rerank.recall[k] /= n;

    agg.rrf.ndcg[k] /= n;
    agg.rerank.ndcg[k] /= n;
  }
  agg.rrf.mrr /= n;
  agg.rerank.mrr /= n;

  // Format values between 0 and 1 as percentages.
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  console.log("\nRetrieval evaluation (higher is better)\n");

  // Display RRF and reranked results side by side.
  // This makes it easy to see whether the reranker
  // improved the retrieval quality.
  console.log(
    `${"metric".padEnd(10)} | ${"RRF (pre-rerank)".padEnd(18)} | ${"Reranked".padEnd(18)}`,
  );
  console.log("-".repeat(54));

  // Display Recall@K for each cutoff.
  for (const k of ks) {
    console.log(
      `R@${k}`.padEnd(10) +
        " | " +
        pct(agg.rrf.recall[k]).padEnd(18) +
        " | " +
        pct(agg.rerank.recall[k]).padEnd(18),
    );
  }

  // Display Mean Reciprocal Rank.
  console.log(
    "MRR".padEnd(10) +
      " | " +
      agg.rrf.mrr.toFixed(3).padEnd(18) +
      " | " +
      agg.rerank.mrr.toFixed(3).padEnd(18),
  );

  // Display nDCG@K for each cutoff.
  for (const k of ks) {
    console.log(
      `nDCG@${k}`.padEnd(10) +
        " | " +
        agg.rrf.ndcg[k].toFixed(3).padEnd(18) +
        " | " +
        agg.rerank.ndcg[k].toFixed(3).padEnd(18),
    );
  }
  console.log();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/**
 * Runs the retrieval evaluation.
 *
 * The evaluation supports two optional modes:
 *
 * `--no-ingest`
 * - Skip indexing the sample documents.
 * - Useful when the evaluation dataset is already indexed.
 *
 * `--fresh`
 * - Delete the existing Vector and BM25 indexes first.
 * - Useful for running a clean benchmark without documents
 *   from previous experiments affecting the results.
 */
async function main() {
  // Check which command-line options were provided.
  const skipIngest = process.argv.includes("--no-ingest");
  const fresh = process.argv.includes("--fresh");

  // Get the document IDs used by the evaluation dataset.
  // Use their filenames as `docId`
  const evalDocIds = new Set(
    (await fs.readdir(SAMPLE_DIR)).filter((f) => f.endsWith(".md")),
  );

  // Start with completely empty indexes when `--fresh` is used.
  if (fresh) {
    console.log("Fresh mode: wiping index…");

    // Delete the LanceDB vector index.
    await fs.rm(path.join(process.cwd(), "data", "lancedb"), {
      recursive: true,
      force: true,
    });

    // Delete the MiniSearch BM25 index.
    await fs.rm(path.join(process.cwd(), "data", "minisearch.json"), {
      force: true,
    });
  }

  // Index the sample documents unless `--no-ingest` was provided
  if (!skipIngest) {
    console.log("Ingesting sample document(s)…");
    await ingestSampleDocs(evalDocIds);
  } else {
    // When ingestion is skipped, make sure an index already exists
    const sources = await getVectorSources();
    if (sources.length === 0) {
      console.error(
        "Index is empty. Run `npm run eval` (without --no-ingest) first, or upload documents via the UI.",
      );
      process.exit(1);
    }
  }

  // Run the actual retrieval evaluation.
  await evaluate(evalDocIds);
}

main().catch((err) => {
  console.error("Evaluation failed:", err);
  process.exit(1);
});
