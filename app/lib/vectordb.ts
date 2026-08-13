import * as lancedb from "@lancedb/lancedb";
import path from "path";
import { Chunk, SearchResult } from "./types";

type LanceSearchRow = {
  id: string;
  text: string;
  source: string;
  _distance: number;
};

let db: lancedb.Connection | null = null;
const TABLE_NAME = "chunks";

// Use an absolute path so every process (and every hot-reloaded route
// handler) resolves the dataset at exactly the same location. A relative path
// here lets the on-disk manifests store data-file paths relative to a base
// that can differ between the process that wrote a table and later readers,
// which surfaces as hard-to-debug "Not found .../<file>.lance" errors.
const DB_PATH = path.join(process.cwd(), "data", "lancedb");

async function getDB(): Promise<lancedb.Connection> {
  if (!db) {
    db = await lancedb.connect(DB_PATH);
  }
  return db;
}

async function tableExists(connection: lancedb.Connection): Promise<boolean> {
  const names = await connection.tableNames();
  return names.includes(TABLE_NAME);
}

export async function addChunks(
  chunks: Chunk[],
  vectors: number[][],
): Promise<void> {
  const connection = await getDB();
  const data = chunks.map((chunk, i) => ({
    id: chunk.id,
    text: chunk.text,
    source: chunk.source,
    vector: vectors[i],
  }));

  if (await tableExists(connection)) {
    await connection.openTable(TABLE_NAME).then((t) => t.add(data));
  } else {
    // No dummy-row + delete() dance: create the table directly from real data.
    await connection.createTable(TABLE_NAME, data);
  }
}

export async function searchSimilar(
  queryVector: number[],
  topK = 5,
): Promise<SearchResult[]> {
  const connection = await getDB();
  // Nothing indexed yet — do not build/read an empty table.
  if (!(await tableExists(connection))) {
    return [];
  }
  const tbl = await connection.openTable(TABLE_NAME);
  const results = (await tbl
    .search(queryVector)
    .limit(topK)
    .toArray()) as LanceSearchRow[];
  return results.map((row) => ({
    text: row.text,
    source: row.source,
    score: row._distance,
  }));
}
