export interface Chunk {
  id: string;
  text: string;
  source: string;
  chunkIndex: number;
}

export interface SearchResult {
  id: string;
  text: string;
  source: string;
  score: number;
}

// A single raw entry shared by both the vector store and the BM25 index.
export interface IndexDocument {
  id: string;
  text: string;
  source: string;
}
