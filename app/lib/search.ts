import { searchBM25 } from "./minisearch";
import { SearchResult } from "./types";
import { searchSimilar } from "./vectordb";

/**
 * Reciprocal Rank Fusion (RRF)
 *
 * RRF combines results from different search methods using their
 * rankings instead of their original scores.
 *
 * We use:
 *
 * 1. Vector Search
 *    - Finds documents with similar meaning.
 *
 * 2. BM25 Search
 *    - Finds documents with matching keywords.
 *
 * Why RRF?
 *
 * Vector search and BM25 produce different types of scores,
 * so their scores cannot be directly compared.
 *
 * RRF avoids this problem by looking only at the position
 * (rank) of each document in each search result.
 *
 * A document that ranks highly in both searches receives
 * a higher combined score.
 *
 * Formula:
 *
 *   RRF Score = 1 / (k + rank)
 *
 * `k` controls how strongly rank affects the score.
 * 60 is a commonly used value.
 */
const RRF_K = 60;

// Fetch extra results from each search method before combining them.
// This gives RRF more candidates to work with.
const CANDIDATES = 20;

/**
 * Performs hybrid search using Vector Search + BM25 + RRF.
 */
export async function searchHybrid(
  query: string,
  queryVector: number[],
  topK = 5,
): Promise<SearchResult[]> {
  // Run both searches at the same time.
  const fetchSize = Math.max(CANDIDATES, topK);
  const [vectorResults, bm25Results] = await Promise.all([
    searchSimilar(queryVector, fetchSize),
    searchBM25(query, fetchSize),
  ]);

  // Store the combined RRF score for each document.
  const scores = new Map<string, number>();

  // Stores the actual document data.
  // The document ID is used to remove duplicates.
  const documents = new Map<string, SearchResult>();

  /**
   * Add Vector Search rankings to the RRF score.
   *
   * `index + 1` gives us the document's rank.
   *
   * Example:
   * index 0 -> rank 1
   * index 1 -> rank 2
   * index 2 -> rank 3
   */
  vectorResults.forEach((result, index) => {
    // Higher-ranked results get a larger RRF contribution.
    const rank = index + 1;
    const rrfScore = 1 / (RRF_K + rank);

    // If this document already has a score from another search,
    // add this score to it.
    scores.set(result.id, (scores.get(result.id) ?? 0) + rrfScore);

    // Keep the document so we can return it later.
    documents.set(result.id, result);
  });

  /**
   * Add BM25 rankings to RRF.
   *
   * If the same document was also found by Vector Search,
   * its RRF score is added to the existing score.
   */
  bm25Results.forEach((result, index) => {
    const rank = index + 1;
    const rrfScore = 1 / (RRF_K + rank);

    scores.set(result.id, (scores.get(result.id) ?? 0) + rrfScore);

    documents.set(result.id, result);
  });

  /**
   * Create the final ranked result list.
   *
   * Documents with higher RRF scores are more relevant.
   */
  return [...documents.entries()]
    .map(([id, document]) => ({
      ...document,
      score: scores.get(id) ?? 0,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
