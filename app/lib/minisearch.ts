import { promises as fs } from "fs";
import MiniSearch from "minisearch";
import path from "path";
import { IndexDocument, SearchResult } from "./types";

interface PersistedStore {
  documents: IndexDocument[];
}

// Store the BM25 index data in a JSON file.
// This means our search index is not lost when the server restarts.
const INDEX_PATH = path.join(process.cwd(), "data", "minisearch.json");

// Keep the search index in memory after loading it.
// This prevents us from rebuilding the index for every search.
let miniServer: MiniSearch | null = null;

// Keep a list of the documents that have been added to the index.
let documents: IndexDocument[] | null = null;

// Number of BM25 results to return by default.
const DEFAULT_BM25_TOP_K = 10;

/**
 * Creates MiniSearch index.
 *
 * BM25 is a keyword-based search algorithm.
 * It looks at things like:
 * - Which words from the query appear in the document
 * - How often those words appear
 * - How important those words are in the whole collection
 */
function createIndex(): MiniSearch {
  return new MiniSearch({
    // Search only inside the "text" field
    fields: ["text"],

    // Use the document's "id" as its unique identifier.
    idField: "id",

    // These fields will be available when we get search results.
    storeFields: ["id", "text", "source", "chunkIndex", "docId"],

    // Break text into individual words.
    tokenize: (text) => text.split(/[\s\-.,!?;:()"']+/),

    searchOptions: {
      // Allow partial word matching.
      // Example: "compute" can match "computer".
      prefix: true,

      // BM25 settings.
      // These control how MiniSearch calculates the relevance score.
      bm25: {
        k: 1.2,
        b: 0.75,
        d: 0.5,
      },
    },
  });
}

/**
 * Loads the BM25 index.
 *
 * The first time this function runs:
 * 1. Create a new MiniSearch index.
 * 2. Load previously saved documents from disk.
 * 3. Add those documents to the index.
 *
 * After that, reuse the same index from memory.
 */
async function ensureIndex(): Promise<{
  mini: MiniSearch;
  indexDocuments: IndexDocument[];
}> {
  // The index is already loaded, so reuse it.
  if (miniServer && documents) {
    return {
      mini: miniServer,
      indexDocuments: documents,
    };
  }

  const mini = createIndex();
  let loaded: IndexDocument[] = [];

  try {
    // Try to load previously indexed documents from disk.
    const raw = await fs.readFile(INDEX_PATH, "utf-8");
    const persisted = JSON.parse(raw) as PersistedStore;
    loaded = persisted.documents ?? [];

    // Add documents to the index in batches.
    // This is easier on memory when there are many documents.
    for (let i = 0; i < loaded.length; i += 1000) {
      mini.addAll(loaded.slice(i, i + 1000));
    }
  } catch {
    // The file doesn't exist yet.
    // We'll start with an empty index.
  }

  // Save the index in memory for future searches.
  miniServer = mini;
  documents = loaded;

  return {
    mini: miniServer,
    indexDocuments: documents,
  };
}

/**
 * Saves the indexed documents to disk.
 *
 * We save the documents so we can rebuild the BM25 index
 * when the server starts again.
 */
async function persist(docs: IndexDocument[]): Promise<void> {
  const store: PersistedStore = { documents: docs };

  // Make sure the data directory exists.
  await fs.mkdir(path.dirname(INDEX_PATH), { recursive: true });

  // Save the documents as JSON.
  await fs.writeFile(INDEX_PATH, JSON.stringify(store));
}

/**
 * Adds document chunks to the BM25 index.
 */
export async function addChunksToIndex(chunks: IndexDocument[]): Promise<void> {
  // Nothing to add.
  if (chunks.length === 0) return;

  // Get the IDs of documents that are already indexed.
  // This prevents us from adding the same document twice.
  const { mini } = await ensureIndex();

  // MiniSearch rejects duplicate ids; skip anything we have already indexed.
  const existing = new Set(documents!.map((d) => d.id));

  // Keep only documents that haven't been indexed yet.
  const fresh = chunks
    .filter((c) => !existing.has(c.id))
    .map(
      ({ id, docId, text, source, chunkIndex }): IndexDocument => ({
        id,
        docId,
        text,
        source,
        chunkIndex,
      }),
    );

  if (fresh.length > 0) {
    // Add the new documents to the BM25 index.
    mini.addAll(fresh);

    // Also keep them in our document list.
    documents = documents!.concat(fresh);

    // Save the updated documents to disk.
    await persist(documents);
  }
}

/**
 * Searches the BM25 index using a text query.
 *
 * BM25 finds documents that contain words related to the query.
 */
export async function searchBM25(
  query: string,
  topK = DEFAULT_BM25_TOP_K,
): Promise<SearchResult[]> {
  const { mini } = await ensureIndex();

  // No documents have been indexed yet.
  if (mini.documentCount === 0) return [];

  // Search the index using the user's query.
  //
  // prefix: true
  // Allows partial word matching.
  //
  // fuzzy: 0.2
  // Allows small spelling differences.
  const hits = mini.search(query, { prefix: true, fuzzy: 0.2 });

  // Return only the best matching results.
  return hits.slice(0, topK).map((hit) => ({
    id: String(hit.id),
    text: hit.text,
    source: hit.source,
    chunkIndex: (hit as unknown as { chunkIndex?: number }).chunkIndex ?? 0,
    score: hit.score,
  }));
}

/**
 * Returns all documents currently stored in the BM25 index.
 */
export async function getAllIndexedDocuments(): Promise<IndexDocument[]> {
  const { indexDocuments } = await ensureIndex();
  return indexDocuments;
}

/**
 * Removes every chunk that belongs to the given document id.
 *
 * Canonical delete path: a docId uniquely identifies a single
 * indexed document regardless of its file name.
 */
export async function deleteIndexByDocId(docId: string): Promise<void> {
  const { mini, indexDocuments } = await ensureIndex();

  const toRemove = indexDocuments.filter((d) => d.docId === docId);
  if (toRemove.length === 0) return;

  // removeAll expects the full document objects, not just IDs.
  mini.removeAll(toRemove);

  documents = indexDocuments.filter((d) => d.docId !== docId);
  await persist(documents);
}

/**
 * Returns the distinct documents currently indexed,
 * along with how many chunks each one has.
 *
 * Identity is the document id (docId); the source file name is
 * only used for display.
 */
export async function getSources(): Promise<
  { docId: string; source: string; chunks: number }[]
> {
  const { indexDocuments } = await ensureIndex();

  const counts = new Map<string, { source: string; chunks: number }>();
  for (const doc of indexDocuments) {
    const existing = counts.get(doc.docId);
    if (existing) {
      existing.chunks += 1;
    } else {
      counts.set(doc.docId, { source: doc.source, chunks: 1 });
    }
  }

  return [...counts.entries()].map(([docId, { source, chunks }]) => ({
    docId,
    source,
    chunks,
  }));
}
