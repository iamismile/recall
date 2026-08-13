export interface Chunk {
  id: string;
  text: string;
  source: string;
  chunkIndex: number;
}

export interface SearchResult {
  text: string;
  source: string;
  score: number;
}
