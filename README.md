# Recall — Local-First RAG Memory

**Recall** is a personal semantic memory app: upload your documents (`.txt`, `.md`, `.pdf`), then ask questions in natural language. It retrieves the most relevant passages and uses an LLM to generate a grounded answer with citations.

Built to demonstrate an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline with a **local-first** design — embeddings and search run entirely on your machine; only the final answer generation calls an external model.

## Architecture

```
Upload ─▶ Parse ─▶ Chunk ─▶ Embed (on-device) ─▶ Store
                                                    ├─ LanceDB   (vector / semantic)
                                                    └─ MiniSearch (BM25 / keyword)
                                                         │
Query ─▶ Embed (on-device) ─▶ Hybrid search (RRF) ─▶ Gemini ─▶ Grounded answer + citations
```

| Stage | Technology | What it does |
|-------|------------|--------------|
| Ingestion | `pdf-parse` | Extract text from PDF / TXT / MD |
| Chunking | custom `chunker.ts` | Overlapping word windows |
| Embeddings | `@huggingface/transformers` (MiniLM) | On-device vectors, no API |
| Vector store | `@lancedb/lancedb` | Semantic similarity search |
| Keyword store | `minisearch` | BM25 lexical search |
| Fusion | Reciprocal Rank Fusion (`search.ts`) | Merges both result lists |
| Generation | Gemini API — Google GenAI SDK (`generate.ts`) | Grounded answer + `[n]` citations |

## Why hybrid search + RRF?

Vector search finds meaning; BM25 finds exact keywords. Their raw scores
aren't comparable, so we rank each list and fuse with
`RRF = 1 / (k + rank)` (k = 60). This beats either method alone and is
robust to queries that are either semantic or lexical.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Create your env file and add a Gemini API key:
   ```bash
   cp .env.example .env.local
   # edit .env.local and set GEMINI_API_KEY
   ```
   Get a key at https://aistudio.google.com/app/apikey
   Uses the Google GenAI SDK (`@google/genai`); default model `gemini-2.5-flash`.
3. Run the dev server:
   ```bash
   npm run dev
   ```
4. Open http://localhost:3000, upload a document, and ask a question.

## API

- `POST /api/upload` — upload a file (form-data `file`). Upserts by filename.
- `POST /api/search` — `{ query }` → **Server-Sent Events** stream: `sources` (retrieved chunks, sent first), `token` (answer text deltas), `error` (generation failure), `done`.
- `GET  /api/documents` — list indexed documents (`docId`, source, chunks).
- `DELETE /api/documents?docId=…` — delete a document and all its chunks.

## Notes

- Documents are identified by a `docId` (not filename) so updates/deletes are
  collision-free.
- Indexes persist locally in `data/`; embedding models cache in `.cache/`.
- Generation requires `GEMINI_API_KEY`; retrieval works fully offline.
