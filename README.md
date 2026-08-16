# Recall — Local-First RAG Memory

**Recall** is a personal semantic memory app: upload your documents (`.txt`, `.md`, `.pdf`), then ask questions in natural language. It retrieves the most relevant passages and uses an LLM to generate a grounded answer with citations.

This project is also a **worked example of Retrieval-Augmented Generation (RAG)**. The README is written to teach the core ideas of RAG — what each stage is, why it exists, and how it is implemented here — so you can read it as a tutorial and then explore the code.

---

## What is RAG?

Large language models (LLMs) are great at reasoning over text, but they only "know" what they learned during training. They have no access to **your** files, and they can confidently make things up (*hallucination*).

**Retrieval-Augmented Generation** fixes this by splitting the job in two:

1. **Retrieve** — given the user's question, find the most relevant pieces of text from a knowledge base.
2. **Generate** — feed those retrieved pieces to the LLM as *context*, and ask it to answer **using only that context**, citing its sources.

```
            ┌─────────────────────────────────────────────┐
 Question ─▶│  Retrieve the right context                 │
            │        (search + rank your documents)       │
            └───────────────────┬─────────────────────────┘
                                │  relevant passages
                                ▼
            ┌─────────────────────────────────────────────┐
            │  Generate an answer grounded in those passages│
            │        (LLM + citations)                     │
            └─────────────────────────────────────────────┘
```

The result is an answer that is **grounded in your data**: the model can only say things supported by retrieved text, and it tells you *where* each claim came from.

RAG has two phases:

- **Ingestion (offline):** turn your documents into a searchable index. This happens once per document (on upload).
- **Retrieval + Generation (online):** run on every question.

Recall makes the ingestion and retrieval steps run **locally on your machine**; only the final answer generation calls an external model (Gemini).

---

## The RAG pipeline in Recall

```
Upload ─▶ Parse ─▶ Chunk ─▶ Embed (on-device) ─▶ Store
                                                     ├─ LanceDB   (vector / semantic)
                                                     └─ MiniSearch (BM25 / keyword)
                                                          │
Query ─▶ Embed (on-device) ─▶ Hybrid search (RRF, top 20) ─▶ Rerank (cross-encoder) ─▶ Gemini ─▶ Grounded answer + citations
```

| Stage | Technology | What it does |
|-------|------------|--------------|
| Ingestion | `pdf-parse` | Extract text from PDF / TXT / MD |
| Chunking | custom `chunker.ts` | Overlapping word windows |
| Embeddings | `@huggingface/transformers` (MiniLM) | On-device vectors, no API |
| Vector store | `@lancedb/lancedb` | Semantic similarity search |
| Keyword store | `minisearch` | BM25 lexical search |
| Fusion | Reciprocal Rank Fusion (`search.ts`) | Merges both result lists → top 20 candidates |
| Reranking | Local cross-encoder (`rerank.ts`, on-device) **or** Jina Reranker API | Re-scores candidates together with the query → top 5 |
| Generation | Gemini API — Google GenAI SDK (`generate.ts`) | Grounded answer + `[n]` citations |

The next sections explain each concept in plain language, then point to the file that implements it.

---

## Ingestion: getting documents into the system

### 1. Parsing — "get the text out"
A file is not plain text. A PDF stores characters as positioned glyphs; Markdown has markup; a `.txt` is just UTF-8. The first step is to normalize every supported format into raw text.

- `app/lib/parsers.ts` — `extractTextFromFile()` reads `.txt`/`.md` directly and runs PDFs through `pdf-parse`.
- `app/api/upload/route.ts` — the upload endpoint that saves the file, parses it, and rejects empty/unsupported files.

### 2. Chunking — "don't search the whole book at once"
We don't store or search a document as one block of text. If you ask *"What does it say about JWT?"* and the answer is in page 12 of a 200-page PDF, a single giant embedding for the whole file will be too diluted to match well.

So we split the text into **chunks** — small, searchable windows.

- `app/lib/chunker.ts` — `chunkText()` splits into windows of `chunkSize` words (default **200**) with `overlap` words (default **20**) shared between neighbors.
- **Why overlap?** A sentence that falls exactly on a boundary would be split across two chunks and lose meaning. Overlap keeps context continuous so a match is never "cut in half."

> Concept: **chunk size is a trade-off.** Too small → loses surrounding context; too large → matches become fuzzy. Overlap softens that trade-off. Recall uses 200/20 as a sensible default.

Every chunk inherits a single `docId` for the whole upload, so a document can later be deleted or re-uploaded as one unit (see `app/api/upload/route.ts`).

### 3. Embeddings — "turn meaning into math"
An embedding model converts text into a **vector** (a list of numbers) that captures *meaning*. Texts with similar meaning land close together in vector space, even if they use different words.

- `app/lib/embeddings.ts` — uses `Xenova/all-MiniLM-L6-v2` via transformers.js, running **fully on-device** (no API call).
- `pooling: "mean"` averages token vectors into one vector per chunk; `normalize: true` makes all vectors comparable by cosine/dot-product similarity.

> Concept: **bi-encoder.** The embedding model encodes the query and each chunk *independently*. That's what makes retrieval fast (you can pre-compute all chunk vectors once and store them), but it sees query and passage in isolation, which limits precision. We fix that later with a cross-encoder (reranking).

### 4. Storing — two indexes for two kinds of search
Recall builds **two complementary indexes** over the same chunks:

- **Vector store — `app/lib/vectordb.ts`** (LanceDB, persisted in `data/lancedb`). Stores each chunk's embedding for *semantic* search.
- **Keyword store — `app/lib/minisearch.ts`** (MiniSearch, persisted in `data/minisearch.json`). Stores the raw text for *lexical* (BM25) search.

> Concept: **semantic vs. lexical.** Vector search answers *"which passage means the same thing?"* BM25 answers *"which passage literally contains these words?"* A query like "How do I reset my password?" matches semantically; a query like "ErrorCode 0x7F3A" matches lexically. Neither is enough alone — which is why we use both (hybrid search, below).

---

## Retrieval: answering a question

### 5. The two search methods: semantic search vs. BM25

Before fusing results, it helps to understand what the two searches actually do. They answer the same question — *"which chunks are relevant to this query?"* — in completely different ways.

#### Semantic search (vector search) — "find meaning, not words"
Imagine every chunk of text is plotted as a **point** in a giant space, where points that mean similar things sit close together. When you ask a question, we plot your question as a point too, then look for the nearest chunk-points. That "nearness" is what `searchSimilar` in `app/lib/vectordb.ts` computes over the LanceDB embeddings.

- **How it works (in one breath):** the embedding model (`embeddings.ts`) turns text into a vector; similar meanings → similar vectors; we rank chunks by vector distance to the query.
- **What it gives us:** it matches on *meaning*. You can ask *"How do I recover a lost password?"* and it will find a chunk titled *"Resetting your account credentials"* — even though the words barely overlap. It handles synonyms, paraphrasing, and loosely-worded questions.
- **Where it's weak:** "meaning" is fuzzy. It can miss a literal, specific token — an error code like `0x7F3A`, a product SKU, a person's name, or an exact phrase — because those aren't about *meaning*, they're about *exact characters*.

#### BM25 (keyword / lexical search) — "find the exact words"
BM25 is the classic search-engine algorithm (the "does this page contain my words?" approach) used by `app/lib/minisearch.ts`. It scores each chunk by:

- **Term Frequency (TF):** how often your query words appear in the chunk — more appearances → more relevant.
- **Inverse Document Frequency (IDF):** rare words (like `0x7F3A`) count *more* than common words (like `the` or `and`), because rare words are more telling.
- **Length normalization:** a match in a short chunk counts more than the same match buried in a huge chunk.

- **How it works (in one breath):** it counts and weights word overlaps between query and chunk, favoring rare, repeated, and well-matched terms.
- **What it gives us:** precise matches for **exact terms** — error codes, IDs, names, technical strings, or when a user pastes a literal phrase. If you search `ErrorCode 0x7F3A`, BM25 nails it.
- **Where it's weak:** it doesn't understand meaning. It won't connect *"car"* to *"automobile"*, or a paraphrased question to a differently-worded answer.

#### Why we use both (hybrid)
Neither method is "better" — they cover each other's blind spots:

| | Semantic search | BM25 |
|---|---|---|
| Best at | *Meaning*, intent, synonyms | *Exact* words, codes, names |
| Struggles with | Specific literal tokens | Paraphrase, synonyms |
| Example query | "How do I recover a lost password?" | `ErrorCode 0x7F3A` |

Running both means a question is answered whether the user *described the concept* (semantic) or *typed the exact string* (BM25). The next stage combines their strengths.

### 6. Hybrid search + Reciprocal Rank Fusion (RRF)
At query time, Recall:

1. Embeds the query (`embedTexts([query])`).
2. Runs **both** searches in parallel — vector (LanceDB) and BM25 (MiniSearch) — pulling the top 20 candidates each.
3. Fuses the two ranked lists with **Reciprocal Rank Fusion**.

- `app/lib/search.ts` — `searchHybrid()` and the RRF implementation.

**Why RRF?** Vector search and BM25 produce scores on totally different scales (distances vs. term-frequency weights), so you can't just add them. RRF ignores the raw scores and uses only **rank position**:

```
RRF_score = 1 / (k + rank)      k = 60
```

A chunk that ranks highly in *both* lists gets a bigger combined score. RRF is simple, robust, and needs no tuning of score weights.

> Concept: **retrieve-then-rerank.** RRF is a cheap "first pass" that produces a shortlist of ~20 candidates. It's good but coarse — it only knows *where* a passage ranked, not *how* relevant it actually is to this specific query. The next stage sharpens it.

### 7. Reranking — "read query and passage together"
The top ~20 fused candidates are re-scored by a **cross-encoder**: a model that takes the query *and* a passage as a **pair** and outputs a single relevance score (0–1).

- `app/lib/rerank.ts` — `rerankChunks()`; keeps the top 5.

> Concept: **bi-encoder vs. cross-encoder.**
> - *Bi-encoder* (embedding model): encodes query and passage separately → fast, pre-computable, but shallow.
> - *Cross-encoder* (reranker): encodes them **jointly** → much more accurate, but must run live on every candidate.
>
> So the standard production pattern is: cheap bi-encoder + RRF to get ~20 candidates, then an expensive cross-encoder to pick the best 5. That gives you both speed and precision.

The reranker is **pluggable**:

- `RERANK_PROVIDER=local` (default) — runs `Xenova/ms-marco-MiniLM-L-6-v2` on-device via transformers.js. No external calls; privacy-first. Loads on first query, then cached in `.cache/`.
- `RERANK_PROVIDER=jina` — calls the **Jina Reranker API** instead, so no heavy model is loaded into our own process (ideal for serverless or memory-constrained hosts). Requires `JINA_RERANKING_API_KEY`.

> If reranking fails (e.g. model load error), `app/api/search/route.ts` falls back to the RRF ranking so search still works.

### 8. Generation — "answer from the context, and cite it"
The top 5 passages become the **context** for the LLM. Recall sends them as a numbered list and instructs the model to:

- answer using **only** the provided context,
- cite each claim with its marker (`[1]`, `[2]`, …),
- say *"I don't have that information in your documents."* when the answer isn't present.

- `app/lib/generate.ts` — `buildPrompt()` constructs the grounded prompt; `streamAnswer()` streams tokens from Gemini (`gemini-2.5-flash` by default).
- `app/api/search/route.ts` — returns the answer as a **Server-Sent Events (SSE)** stream: `sources` first (the retrieved chunks), then `token` deltas, then `done` (or `error`).

> Concept: **grounding + citations.** Tying every claim to a retrieved passage is what makes RAG trustworthy. The `[n]` markers let the UI show the user exactly which chunk supported each sentence, and let the model abstain when the data doesn't contain an answer.

---

## Project structure (where to read next)

```
app/
├── api/
│   ├── upload/route.ts        # ingestion endpoint: parse → chunk → embed → store
│   ├── search/route.ts        # query endpoint: embed → hybrid → rerank → stream answer
│   └── documents/route.ts     # list / delete indexed documents
├── components/                # UI: FileUpload, DocumentManager, SearchBar, Results, Answer
└── lib/
    ├── parsers.ts             # text extraction (.txt/.md/.pdf)
    ├── chunker.ts             # overlapping word-window chunking
    ├── embeddings.ts          # on-device MiniLM embeddings
    ├── vectordb.ts            # LanceDB vector store
    ├── minisearch.ts          # MiniSearch BM25 index
    ├── search.ts              # hybrid search + RRF
    ├── rerank.ts              # cross-encoder reranking (local / jina)
    ├── generate.ts            # Gemini grounded prompt + streaming
    └── types.ts               # Chunk / SearchResult / IndexDocument
```

The `lib/` folder maps almost one-to-one onto the RAG stages above, which makes it a good place to study each concept in isolation.

---

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Create your env file and add a Gemini API key:
   ```bash
   cp .env.example .env
   # edit .env and set GEMINI_API_KEY
   ```
   Get a key at https://aistudio.google.com/app/apikey
   Uses the Google GenAI SDK (`@google/genai`); default model `gemini-2.5-flash`.
3. Run the dev server:
   ```bash
   npm run dev
   ```
4. Open http://localhost:3000, upload a document, and ask a question.

## Configuration

All options live in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Required for answer generation. Retrieval works without it. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Override the generation model. |
| `RERANK_PROVIDER` | `local` | `local` (on-device cross-encoder) or `jina` (API). |
| `JINA_RERANKING_API_KEY` | — | Required only when `RERANK_PROVIDER=jina`. |
| `JINA_RERANK_MODEL` | `jina-reranker-v2-base-multilingual` | Jina model to use. |

## API

- `POST /api/upload` — upload a file (form-data `file`). Upserts by filename.
- `POST /api/search` — `{ query }` → **Server-Sent Events** stream: `sources` (retrieved chunks, sent first), `token` (answer text deltas), `error` (generation failure), `done`.
- `GET  /api/documents` — list indexed documents (`docId`, source, chunks).
- `DELETE /api/documents?docId=…` — delete a document and all its chunks.

## Notes

- Documents are identified by a `docId` (not filename) so updates/deletes are collision-free.
- Indexes persist locally in `data/`; embedding models cache in `.cache/`.
- Generation requires `GEMINI_API_KEY`; retrieval works fully offline.
- Reranking loads a second on-device model (`ms-marco-MiniLM-L-6-v2`) on first query when `RERANK_PROVIDER=local` (the default); it is cached in `.cache/` afterwards. Set `RERANK_PROVIDER=jina` to skip local model loading.
