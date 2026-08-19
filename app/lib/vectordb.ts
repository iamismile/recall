import * as lancedb from "@lancedb/lancedb";
import path from "path";
import { Chunk, SearchResult } from "./types";

type LanceSearchRow = {
  id: string;
  docId: string;
  text: string;
  source: string;
  chunkIndex: number;
  _distance: number;
};

// Store the LanceDB database inside our project's data folder.
const DB_PATH = path.join(process.cwd(), "data", "lancedb");

// Keep the database connection in memory so we don't reconnect every time.
let db: lancedb.Connection | null = null;

// Name of the table where we store document chunks and their vectors.
const TABLE_NAME = "chunks";

/**
 * Connects to LanceDB.
 *
 * We create the connection only once and reuse it.
 */
async function getDB(): Promise<lancedb.Connection> {
  if (!db) {
    db = await lancedb.connect(DB_PATH);
  }
  return db;
}

/**
 * Checks whether our "chunks" table already exists.
 */
async function tableExists(connection: lancedb.Connection): Promise<boolean> {
  const names = await connection.tableNames();
  return names.includes(TABLE_NAME);
}

/**
 * Adds document chunks and their embedding vectors to LanceDB.
 *
 * Each chunk contains:
 * - id: Unique ID of the chunk
 * - text: The actual text
 * - source: Where the text came from
 * - vector: The embedding of the text
 */
export async function addChunks(
  chunks: Chunk[],
  vectors: number[][],
): Promise<void> {
  const connection = await getDB();

  // Combine the text information with its embedding vector.
  const data = chunks.map((chunk, i) => ({
    id: chunk.id,
    docId: chunk.docId,
    text: chunk.text,
    source: chunk.source,
    chunkIndex: chunk.chunkIndex,
    vector: vectors[i],
  }));

  // If the table already exists, add the new chunks to it.
  if (await tableExists(connection)) {
    const table = await connection.openTable(TABLE_NAME);
    await table.add(data);
  } else {
    // If the table doesn't exist, create it using our first batch of data.
    await connection.createTable(TABLE_NAME, data);
  }
}

/**
 * Searches LanceDB for chunks that are most similar to the query.
 *
 * queryVector:
 * - The embedding vector of the user's search query.
 *
 * topK:
 * - How many results we want to return.
 */
export async function searchSimilar(
  queryVector: number[],
  topK = 5,
): Promise<SearchResult[]> {
  const connection = await getDB();
  // If we haven't indexed anything yet, return an empty result.
  if (!(await tableExists(connection))) {
    return [];
  }

  const tbl = await connection.openTable(TABLE_NAME);

  // Search the database using the query's embedding vector.
  // LanceDB returns the closest matching chunks.
  const results = (await tbl
    .search(queryVector)
    .limit(topK)
    .toArray()) as LanceSearchRow[];

  // Convert LanceDB's result format into our application's format.
  return results.map((row) => ({
    id: row.id,
    docId: row.docId,
    text: row.text,
    source: row.source,
    chunkIndex: row.chunkIndex ?? 0,
    score: row._distance,
  }));
}

/**
 * Removes every chunk that belongs to the given document id.
 *
 * This is the canonical delete path: a docId uniquely identifies a
 * single indexed document regardless of its file name.
 */
export async function deleteByDocId(docId: string): Promise<void> {
  const connection = await getDB();
  if (!(await tableExists(connection))) return;

  const tbl = await connection.openTable(TABLE_NAME);
  await tbl.delete(`docId = '${docId}'`);
}

/**
 * Returns the distinct documents currently stored,
 * along with how many chunks each one has.
 *
 * Identity is the document id (docId); the source file name is
 * only used for display.
 */
export async function getSources(): Promise<
  { docId: string; source: string; chunks: number }[]
> {
  const connection = await getDB();
  if (!(await tableExists(connection))) return [];

  const tbl = await connection.openTable(TABLE_NAME);
  const rows = (await tbl.query().select(["docId", "source"]).toArray()) as {
    docId: string;
    source: string;
  }[];

  const counts = new Map<string, { source: string; chunks: number }>();
  for (const row of rows) {
    const existing = counts.get(row.docId);
    if (existing) {
      existing.chunks += 1;
    } else {
      counts.set(row.docId, { source: row.source, chunks: 1 });
    }
  }

  return [...counts.entries()].map(([docId, { source, chunks }]) => ({
    docId,
    source,
    chunks,
  }));
}
