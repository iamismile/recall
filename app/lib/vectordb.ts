import * as lancedb from "@lancedb/lancedb";
import path from "path";
import { Chunk, SearchResult } from "./types";

type LanceSearchRow = {
  id: string;
  text: string;
  source: string;
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
    text: chunk.text,
    source: chunk.source,
    vector: vectors[i],
  }));

  // If the table already exists, add the new chunks to it.
  if (await tableExists(connection)) {
    await connection.openTable(TABLE_NAME).then((t) => t.add(data));
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
    text: row.text,
    source: row.source,
    score: row._distance,
  }));
}
