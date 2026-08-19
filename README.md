<div align="center">

# Recall

### Your Personal Semantic Memory

A local-first RAG system where you upload documents and ask questions in natural language.

**All retrieval runs on your machine. Only the final answer calls an API.**

[Quick Start](#2-quick-start) · [Learn AI Concepts](#part-2---learn-ai-concepts) · [API Reference](#6-api-reference) · [Experiments](#21-experiment-lab)

---

</div>

> Recall is a Retrieval-Augmented Generation (RAG) application built as an **AI Engineering learning project**. You upload `.txt`, `.md`, or `.pdf` documents, and Recall indexes them so you can search and ask questions about their content using natural language.

### How it works

```
You ask:  "How does authentication work?"
              ↓
Recall searches your documents for relevant passages
              ↓
Gemini generates a grounded answer with citations
              ↓
You get:  "Authentication uses JWT tokens [1]. Configure
           the secret via JWT_SECRET [2]."
```

<div align="center">

| Embeddings | Vector Search | BM25  |  RRF  | Reranking | Generation |
| :--------: | :-----------: | :---: | :---: | :-------: | :--------: |
|   Local    |     Local     | Local | Local |   Local   |   Gemini   |

</div>

---

## Table of Contents

<details open>
<summary><strong>Part 1 - Project & Setup</strong></summary>

- [1. What Is Recall?](#1-what-is-recall)
- [2. Quick Start](#2-quick-start)
- [3. Project Structure](#3-project-structure)
- [4. Technology Stack & Packages](#4-technology-stack--packages)
- [5. Configuration](#5-configuration)
- [6. API Reference](#6-api-reference)
- [7. Evaluation Harness](#7-evaluation-harness)

</details>

<details open>
<summary><strong>Part 2 - Learn AI Concepts</strong></summary>

- [8. What Is RAG?](#8-what-is-rag)
- [9. The Recall Architecture](#9-the-recall-architecture)
- [10. Phase 1 - Document Ingestion](#10-phase-1--document-ingestion)
- [11. Phase 2 - Retrieval](#11-phase-2--retrieval)
- [12. Phase 3 - Reranking](#12-phase-3--reranking)
- [13. Phase 4 - Generation](#13-phase-4--generation)
- [14. Streaming AI Responses](#14-streaming-ai-responses)
- [15. End-to-End Request Flow](#15-end-to-end-request-flow)
- [16. Real Query Walkthrough](#16-real-query-walkthrough)
- [17. Important Trade-offs](#17-important-trade-offs)
- [18. RAG Failure Modes](#18-rag-failure-modes)
- [19. Evaluating a RAG System](#19-evaluating-a-rag-system)
- [20. How to Study This Project](#20-how-to-study-this-project)
- [21. Experiment Lab](#21-experiment-lab)
- [22. Final Mental Model](#22-final-mental-model)

</details>

---

# Part 1 - Project & Setup

## 1. What Is Recall?

Recall is a personal knowledge application. You upload documents, and Recall indexes them so you can search and ask questions about their content.

When you ask a question like:

```
"How does authentication work in this project?"
```

Recall:

1. **Searches** your indexed documents for relevant passages
2. **Ranks** the most relevant passages
3. **Passes** those passages to Gemini as context
4. **Generates** an answer based on that context
5. **Cites** its sources so you can verify

> **Design principle:** The LLM should not be responsible for remembering your documents. Retrieval supplies the knowledge; the LLM performs reasoning over that knowledge.

---

## 2. Quick Start

### Prerequisites

- **Node.js 18+** (recommended: 20+)
- **npm** (or pnpm)
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- ~2GB free disk space (for model downloads on first run)

### Install & Run

```bash
git clone <repo-url>
cd recall
npm install
```

Set up your environment:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
```

> That's the only required key. Everything else runs locally.

Start the dev server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), upload a document, and ask a question.

### First Run Notes

On the first upload or query, Recall downloads ML models to the `.cache/` directory (~180MB total). This only happens once, models are cached for subsequent runs.

| Component           | Behavior                                                                                                              |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------- |
| **Embedding model** | Downloads `all-MiniLM-L6-v2` (~90MB) on first use. Runs entirely on your machine.                                     |
| **Reranking model** | If `RERANK_PROVIDER=local` (default), downloads `ms-marco-MiniLM-L-6-v2` (~90MB). Set `RERANK_PROVIDER=jina` to skip. |
| **Vector database** | LanceDB persists to `data/lancedb/`                                                                                   |
| **Keyword index**   | MiniSearch persists to `data/minisearch.json`                                                                         |

<details>
<summary><strong>Troubleshooting</strong></summary>

**"GEMINI_API_KEY not found"** - Check that `.env` exists and contains your key. Restart the dev server after editing.

**Model download fails** - The first query requires internet access to download models from HuggingFace. Check your connection and retry.

**Search returns "I don't have that information in your documents."** - This is expected when no relevant chunk is found. Upload a document with content matching your question, or ask something different.

**PDF parsing returns no text** - Some PDFs are image-based (scanned). Recall requires text-based PDFs. Try a `.txt` or `.md` file instead.

</details>

---

## 3. Project Structure

```
recall/
├── app/
│   ├── api/
│   │   ├── upload/route.ts        # POST /api/upload — document ingestion
│   │   ├── search/route.ts        # POST /api/search — query with SSE streaming
│   │   └── documents/route.ts     # GET/DELETE /api/documents — list & delete
│   │
│   ├── components/                # UI components
│   │   ├── FileUpload.tsx         # File picker with upload logic
│   │   ├── DocumentManager.tsx    # Lists indexed documents, delete capability
│   │   ├── SearchBar.tsx          # Query input form
│   │   ├── Results.tsx            # Displays retrieved chunks with scores
│   │   └── Answer.tsx             # Streamed markdown answer with citations
│   │
│   └── lib/                       # Core AI pipeline (the interesting part)
│       ├── config.ts              # Pipeline constants (chunk size, top-k, etc.)
│       ├── types.ts               # Shared TypeScript types
│       ├── parsers.ts             # Text extraction (.txt/.md/.pdf)
│       ├── chunker.ts             # Overlapping word-window chunking
│       ├── embeddings.ts          # On-device MiniLM embeddings
│       ├── vectordb.ts            # LanceDB vector store
│       ├── minisearch.ts          # MiniSearch BM25 index
│       ├── search.ts              # Hybrid search + Reciprocal Rank Fusion
│       ├── rerank.ts              # Cross-encoder reranking (local or Jina API)
│       └── generate.ts            # Gemini grounded prompt + streaming
│
├── data/                          # Persisted indexes (auto-created, gitignored)
│   ├── lancedb/                   # Vector database
│   └── minisearch.json            # BM25 index
│
├── eval/                          # Offline retrieval evaluation
│   ├── sample-docs/               # 13 benchmark documents
│   └── qa.jsonl                   # 46 gold-standard queries
│
├── scripts/
│   └── evaluate.ts                # Benchmark script (npm run eval)
│
├── .env.example                   # Environment variables template
├── .env                           # Your local configuration (gitignored)
└── package.json
```

### The `app/lib/` Pipeline

Each file in `app/lib/` maps to one stage of the AI pipeline, making it easy to study one concept at a time:

```
parsers.ts       →  Extract text from files
      ↓
chunker.ts       →  Split text into smaller pieces
      ↓
embeddings.ts    →  Convert text chunks into vectors
      ↓
vectordb.ts      →  Store vectors for semantic search
minisearch.ts    →  Store text for keyword search
      ↓
search.ts        →  Find relevant chunks using both methods
      ↓
rerank.ts        →  Reorder chunks by relevance
      ↓
generate.ts      →  Generate an answer using the best chunks
```

---

## 4. Technology Stack & Packages

### Core Framework

| Package               | Version | Purpose                                             |
| :-------------------- | :------ | :-------------------------------------------------- |
| `next`                | 16.3.0  | Full-stack React framework (App Router, API routes) |
| `react` / `react-dom` | 19.2.8  | UI library                                          |
| `typescript`          | ^5      | Type safety (strict mode enabled)                   |

### AI & ML

| Package                         | Version | Purpose                                                 |
| :------------------------------ | :------ | :------------------------------------------------------ |
| `@google/genai`                 | ^2.17.1 | Google Gemini SDK — powers answer generation            |
| `@huggingface/transformers`     | ^4.2.0  | Runs ML models on-device (embeddings + reranking)       |
| `Xenova/all-MiniLM-L6-v2`       | (model) | Embedding model — converts text to 384-dim vectors      |
| `Xenova/ms-marco-MiniLM-L-6-v2` | (model) | Cross-encoder reranker — scores query-passage relevance |

### Data & Search

| Package            | Version | Purpose                                   |
| :----------------- | :------ | :---------------------------------------- |
| `@lancedb/lancedb` | ^0.37.1 | Local vector database for semantic search |
| `minisearch`       | ^7.2.0  | Lightweight BM25/keyword search library   |
| `pdf-parse`        | ^2.4.5  | Extracts text from PDF files              |

### UI & Styling

| Package          | Version | Purpose                                   |
| :--------------- | :------ | :---------------------------------------- |
| `tailwindcss`    | ^4      | Utility-first CSS framework               |
| `react-markdown` | ^10.1.0 | Renders LLM answers as Markdown in the UI |
| `remark-gfm`     | ^4.0.1  | GitHub-Flavored Markdown support          |

### Dev Tools

| Package                         | Version     | Purpose                                  |
| :------------------------------ | :---------- | :--------------------------------------- |
| `eslint` / `eslint-config-next` | ^9 / 16.3.0 | Linting                                  |
| `tsx`                           | ^4.23.12    | TypeScript execution for the eval script |
| `dotenv`                        | ^17.4.2     | Environment variable loading             |

---

## 5. Configuration

All options live in `.env`. Copy `.env.example` to `.env` and adjust as needed.

| Variable                 | Required                       | Default                              | Purpose                                                  |
| :----------------------- | :----------------------------- | :----------------------------------- | :------------------------------------------------------- |
| `GEMINI_API_KEY`         | Yes (for generation)           | —                                    | Google AI Studio API key. Retrieval works without it.    |
| `GEMINI_MODEL`           | No                             | `gemini-2.5-flash`                   | Generation model override                                |
| `RERANK_PROVIDER`        | No                             | `local`                              | `local` = on-device cross-encoder; `jina` = API reranker |
| `JINA_RERANKING_API_KEY` | Only if `RERANK_PROVIDER=jina` | —                                    | Jina API key                                             |
| `JINA_RERANK_MODEL`      | No                             | `jina-reranker-v2-base-multilingual` | Jina model to use                                        |

### Choosing a Reranker Provider

|              | `local` (default)                         | `jina`                               |
| :----------- | :---------------------------------------- | :----------------------------------- |
| **Model**    | `Xenova/ms-marco-MiniLM-L-6-v2`           | `jina-reranker-v2-base-multilingual` |
| **Runs on**  | Your machine                              | Jina's servers                       |
| **Privacy**  | Fully local, no data leaves your machine  | Query + chunks sent to Jina          |
| **Latency**  | No network, but uses CPU/GPU              | Network round-trip                   |
| **Setup**    | Model auto-downloads to `.cache/` (~90MB) | Requires API key                     |
| **Best for** | Privacy-first local use                   | Serverless or weak hardware          |

---

## 6. API Reference

### `POST /api/upload`

Uploads and indexes a document.

**Request**: `multipart/form-data` with a `file` field.

```bash
curl -X POST http://localhost:3000/api/upload \
  -F "file=@my-notes.md"
```

**Pipeline**: `Upload → Parse → Chunk → Embed → Store (LanceDB + MiniSearch)`

**Response**:

```json
{
  "success": true,
  "fileName": "my-notes.md",
  "chunksIndexed": 42
}
```

> **Supported formats:** `.txt`, `.md`, `.pdf` (text-based only)
> **Max file size:** 10 MB
> **Re-uploading** the same filename replaces the old document

---

### `POST /api/search`

Searches indexed documents and streams a grounded answer.

**Request**:

```json
{
  "query": "How does authentication work?"
}
```

**Response**: Server-Sent Events (SSE) stream.

| Event     | Payload               | Description                             |
| :-------- | :-------------------- | :-------------------------------------- |
| `sources` | `Chunk[]`             | Top retrieved chunks (sent first)       |
| `token`   | `string`              | Answer text deltas as they're generated |
| `error`   | `{ message: string }` | Generation failure message              |
| `done`    | —                     | Stream complete                         |

**Pipeline**: `Query → Embed → Hybrid Search (RRF) → Rerank → Gemini → Grounded Answer`

**Example SSE stream**:

```
event: sources
data: [{"id":"chunk-1","text":"...","docId":"a1b2c3","score":0.92}, ...]

event: token
data: The

event: token
data:  application

event: done
```

> **Retrieval happens entirely on-device.** Only the final generation calls Gemini. If reranking fails, the system falls back to the RRF ranking.

---

### `GET /api/documents`

Lists all indexed documents.

```json
[
  {
    "docId": "a1b2c3d4",
    "source": "my-notes.md",
    "chunkCount": 42
  }
]
```

---

### `DELETE /api/documents?docId=...`

Deletes a document and all its chunks from both indexes.

```bash
curl -X DELETE "http://localhost:3000/api/documents?docId=a1b2c3d4"
```

---

## 7. Evaluation Harness

Recall ships with an offline retrieval benchmark that measures whether the retrieval pipeline actually finds the right chunks, **without calling Gemini**.

### Run It

```bash
npm run eval                 # Index sample docs, then evaluate
npm run eval -- --fresh      # Wipe indexes first for an isolated benchmark
npm run eval -- --no-ingest  # Evaluate against already-indexed documents
```

### What It Measures

- **Dataset**: 13 sample documents + 46 gold-standard queries with expected text snippets
- **Relevance**: A retrieved chunk is "relevant" if it contains the expected snippet
- **Metrics**: Recall@K, MRR, nDCG@K, reported for both pre-rerank and post-rerank results

### Sample Output

```
Retrieval evaluation (higher is better)

metric     | RRF (pre-rerank) | Reranked
------------------------------------------------------
R@1        | 55.0%            | 72.0%
R@3        | 78.0%            | 91.0%
R@5        | 84.0%            | 95.0%
MRR        | 0.681            | 0.823
nDCG@5     | 0.701            | 0.874
```

### Why Evaluate Retrieval Separately?

An incorrect final answer doesn't tell you where the system failed:

| Failure Type           | What Happens               | Investigate                                |
| :--------------------- | :------------------------- | :----------------------------------------- |
| **Retrieval failure**  | Wrong chunks found         | Chunking, embeddings, BM25, RRF, reranking |
| **Generation failure** | Right chunks, wrong answer | Prompt construction, LLM behavior          |

The benchmark isolates retrieval, making it cheap, reproducible, and fast; no Gemini calls, just measurable retrieval quality.

---

# Part 2 - Learn AI Concepts

> This section explains the AI engineering concepts used in Recall. Each concept maps directly to a file in the codebase so you can read the code alongside the explanation.

---

## 8. What Is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

A normal LLM works like this:

```
Question  →  LLM  →  Answer
```

The problem: the LLM only knows what was in its training data. It doesn't have access to your private documents.

RAG adds a retrieval step:

```
                     ┌───────────────┐
                     │   Documents   │
                     └───────┬───────┘
                             │
                          Indexing
                             │
                             ▼
Question ──────────────► Retrieval
                             │
                             ▼
                        Relevant Context
                             │
                             ▼
                            LLM
                             │
                             ▼
                          Answer
```

RAG separates the problem into two responsibilities:

| Responsibility | Job                                                  |
| :------------- | :--------------------------------------------------- |
| **Retrieval**  | Find the right information from the user's documents |
| **Generation** | Use that information to produce a useful answer      |

This separation is one of the most important concepts in modern AI applications.

> **Deep dive:** See [Phase 1 — Document Ingestion](#10-phase-1--document-ingestion) for how documents get indexed, and [Phase 2 — Retrieval](#11-phase-2--retrieval) for how searches work.

---

## 9. The Recall Architecture

Recall has two major phases.

### Ingestion Phase (when a document is uploaded)

```
Upload
  ↓
Parse          (extract text from file)
  ↓
Chunk          (split into smaller pieces)
  ↓
Embed          (convert to vectors)
  ↓
Store          (save to LanceDB + MiniSearch)
```

### Query Phase (when the user asks a question)

```
Question
   ↓
Query Embedding
   ↓
┌──────────────────────┐
│                      │
▼                      ▼
Vector Search       BM25 Search
│                      │
└──────────┬───────────┘
           ▼
          RRF            (combine rankings)
           │
       ~20 candidates
           │
           ▼
       Reranking         (cross-encoder re-scores)
           │
        Top 5
           │
           ▼
      Context Builder
           │
           ▼
         Gemini          (generate answer)
           │
           ▼
    Grounded Answer
           │
           ▼
        Citations
```

The implementation maps directly to this architecture:

```
app/lib/
├── parsers.ts        →  Parse
├── chunker.ts        →  Chunk
├── embeddings.ts     →  Embed
├── vectordb.ts       →  Store (vectors)
├── minisearch.ts     →  Store (keywords)
├── search.ts         →  RRF
├── rerank.ts         →  Reranking
└── generate.ts       →  Gemini
```

---

## 10. Phase 1 — Document Ingestion

Before an AI system can retrieve information, that information must first be converted into a searchable representation. This is the ingestion pipeline.

```
File  →  Text  →  Chunks  →  Vectors  →  Indexes
```

### 10.1 Parsing

Different document formats store information differently. A PDF is not simply a string. Markdown contains markup. TXT contains plain text.

The first task is to **normalize different input formats into plain text**.

Recall supports `.txt`, `.md`, and `.pdf` files.

> **Implementation**: `app/lib/parsers.ts`

---

### 10.2 Chunking

**Why can't we embed the entire document?**

Imagine a 200-page PDF. Creating one embedding for the entire document would represent the document as a whole — mixing hundreds of unrelated topics. The relevant information becomes diluted.

Instead, Recall divides documents into smaller pieces called **chunks**:

```
Document
│
├── Chunk 1
├── Chunk 2
├── Chunk 3
├── ...
└── Chunk N
```

Each chunk becomes the basic unit for retrieval.

**Recall's defaults**: 180-word chunks with 20-word overlap.

> **Implementation**: `app/lib/chunker.ts`

### Why overlap?

Important ideas can span chunk boundaries:

```
Chunk A: "The authentication process begins by validating the user's..."

Chunk B: "...JWT token before checking the user's permissions."
```

Without overlap, the connection between these chunks is lost. Overlap preserves some surrounding context:

```
Chunk A
████████████████████
              ██████
              overlap

                    ████████████████████
                    Chunk B
```

### The chunk size trade-off

There is no universally correct chunk size:

|               | Small chunks           | Large chunks           |
| :------------ | :--------------------- | :--------------------- |
| **Retrieval** | More precise           | Less precise           |
| **Context**   | Less context per chunk | More context per chunk |
| **Best for**  | Specific questions     | Broad questions        |

Chunking is an **information retrieval design decision**, not just text preprocessing.

---

### 10.3 Embeddings

An embedding converts text into a vector — a list of numbers that represents the meaning of the text.

For example:

```
"How do I reset my password?"
```

might become something conceptually similar to:

```
[0.12, -0.42, 0.81, 0.07, ...]
```

(The actual vectors have 384 dimensions.)

The important idea:

> **Semantic meaning is represented numerically so that mathematical similarity can be used for retrieval.**

Two pieces of text with similar meanings will have similar vectors, even if they use completely different words.

Recall uses `Xenova/all-MiniLM-L6-v2` running locally via `@huggingface/transformers`.

> **Implementation**: `app/lib/embeddings.ts`

### How similarity is measured

Once texts are converted to vectors, we need a way to compare them. Recall uses **cosine similarity** — a measure of how close two vectors point in the same direction.

A simplified example with 3-dimensional vectors (real ones have 384 dimensions):

```
Query:     "reset my password"
Vector Q = [0.8, 0.2, 0.1]

Chunk A:   "navigate to account settings and change your password"
Vector A = [0.7, 0.3, 0.2]
Cosine similarity(Q, A) = 0.97  ← high (similar meaning)

Chunk B:   "the server uses PostgreSQL for data storage"
Vector B = [0.1, 0.1, 0.9]
Cosine similarity(Q, B) = 0.21  ← low (unrelated topic)
```

Cosine similarity ranges from -1 to 1:

| Score    | Meaning           |
| :------- | :---------------- |
| **1.0**  | Identical meaning |
| **0.0**  | Unrelated         |
| **-1.0** | Opposite meaning  |

The vector database sorts chunks by this similarity score and returns the highest-scoring ones.

### Bi-Encoder (why embeddings are fast)

Recall's embedding model is a **bi-encoder**. The query and document are encoded separately:

```
Query     →  Encoder  →  Query Vector

Document  →  Encoder  →  Document Vector
```

Then similarity is calculated between the vectors.

This is critical for performance because **document embeddings are computed once during ingestion** and stored:

```
During ingestion:
  Document  →  Embedding  →  Store

During query:
  Query  →  Embedding  →  Search stored vectors
```

Without this, every query would need to re-encode every document.

---

### 10.4 Vector Storage

Recall stores embeddings in **LanceDB** — a local, file-based vector database.

At query time:

```
Query embedding  →  Vector similarity search  →  Nearest chunks
```

This is called **approximate nearest neighbor search** — finding chunks with the most similar vectors.

### How a vector search works step by step

When you search for "how do I reset my password":

```
Step 1: Your query is embedded into a vector
        "how do I reset my password" → [0.8, 0.2, 0.1, ...]

Step 2: LanceDB compares this vector against ALL stored chunk vectors
        using cosine similarity

Step 3: Results are sorted by similarity score (highest first)

Step 4: Top matches are returned
```

For example, with 3 chunks in the database:

| Chunk                                | Score | Verdict   |
| :----------------------------------- | :---- | :-------- |
| "To reset your password, go to..."   | 0.97  | Match     |
| "The application uses PostgreSQL..." | 0.21  | Unrelated |
| "File uploads support PDF..."        | 0.35  | Unrelated |

The first chunk wins because its vector is closest to the query vector.

> **Implementation**: `app/lib/vectordb.ts`

### 10.5 Keyword Index

Recall also creates a **BM25 index** using MiniSearch.

Why have two indexes? Because semantic search and keyword search solve different problems:

| Capability  | Semantic Search | Keyword Search (BM25) |
| :---------- | :-------------- | :-------------------- |
| Meaning     | Excellent       | Weak                  |
| Synonyms    | Excellent       | Weak                  |
| Paraphrases | Excellent       | Weak                  |
| Exact terms | Sometimes weak  | Excellent             |
| Error codes | Weak            | Excellent             |
| Names / IDs | Sometimes weak  | Excellent             |

> **Implementation**: `app/lib/minisearch.ts`

---

## 11. Phase 2 — Retrieval

Retrieval is arguably the most important part of a RAG system. A powerful LLM cannot compensate for completely irrelevant context.

> **Bad retrieval → bad context → bad answer.**

### 11.1 Semantic Search

Semantic search asks: **"Which chunks have similar meaning to this question?"**

For example, a query like:

```
"How can I recover my account credentials?"
```

can retrieve a passage about:

```
"To reset your password, navigate to the account settings..."
```

even though the wording is completely different. This is the main advantage of embeddings.

---

### 11.2 BM25 (Keyword Search)

BM25 is a classic information retrieval algorithm. It considers:

| Factor                         | What it measures                                               |
| :----------------------------- | :------------------------------------------------------------- |
| **Term Frequency**             | How often does a query term appear in the document?            |
| **Inverse Document Frequency** | How rare is the term across all documents?                     |
| **Document Length**            | A match in a short passage isn't penalized vs. a long document |

BM25 provides strong lexical matching — it's great for exact terms, error codes, and specific names.

---

### 11.3 Hybrid Search

Semantic search and BM25 have complementary strengths. Using both is more robust than relying on just one:

```
Semantic Search  +  BM25  =  Hybrid Retrieval
```

---

### 11.4 Reciprocal Rank Fusion (RRF)

Now we have two ranked lists with incomparable scores:

```
Semantic search score: 0.82
BM25 score: 13.7
```

You can't add these numbers — they're on completely different scales.

**Reciprocal Rank Fusion (RRF)** solves this by using **rank position** instead of raw scores:

```
RRF_score = 1 / (k + rank)
```

Recall uses `k = 60`.

### Worked example

Suppose we search for "how does authentication work":

**Vector Search results** (ranked by cosine similarity):

| Rank | Chunk                            | Score |
| :--- | :------------------------------- | :---- |
| 1    | "JWT tokens are validated by..." | 0.92  |
| 2    | "The login flow starts with..."  | 0.85  |
| 3    | "File uploads support PDF..."    | 0.31  |

**BM25 results** (ranked by keyword match):

| Rank | Chunk                              | Score |
| :--- | :--------------------------------- | :---- |
| 1    | "The login flow starts with..."    | 14.2  |
| 2    | "JWT tokens are validated by..."   | 11.8  |
| 3    | "Error handling uses try/catch..." | 9.1   |

Notice: the two methods disagree on rank 1 and 2. RRF resolves this.

**RRF calculation** (with k = 60):

```
Chunk: "JWT tokens are validated by..."
  Vector rank = 1  →  1/(60+1) = 0.01639
  BM25 rank   = 2  →  1/(60+2) = 0.01613
  RRF score = 0.01639 + 0.01613 = 0.03252

Chunk: "The login flow starts with..."
  Vector rank = 2  →  1/(60+2) = 0.01613
  BM25 rank   = 1  →  1/(60+1) = 0.01639
  RRF score = 0.01613 + 0.01639 = 0.03252

Chunk: "File uploads support PDF..."
  Vector rank = 3  →  1/(60+3) = 0.01587
  BM25 rank   = —  →  0  (not in BM25 top results)
  RRF score = 0.01587

Chunk: "Error handling uses try/catch..."
  Vector rank = —  →  0  (not in Vector top results)
  BM25 rank   = 3  →  1/(60+3) = 0.01587
  RRF score = 0.01587
```

**Final RRF ranking** (sorted by combined score):

| Chunk                              | RRF Score | Note                          |
| :--------------------------------- | :-------- | :---------------------------- |
| "JWT tokens are validated by..."   | 0.03252   | Ranked by both methods        |
| "The login flow starts with..."    | 0.03252   | Ranked by both methods        |
| "File uploads support PDF..."      | 0.01587   | Only vector search found this |
| "Error handling uses try/catch..." | 0.01587   | Only BM25 found this          |

The key insight: **documents appearing in both lists get boosted**, while documents appearing in only one list get a lower combined score.

RRF acts as a fast **candidate-generation stage** — it finds good candidates cheaply, before the expensive reranking step.

> **Implementation**: `app/lib/search.ts`

---

## 12. Phase 3 — Reranking

After RRF, we have ~20 candidates. But 20 is still too many to send to the LLM, and RRF only knows "this document ranked highly" — it doesn't deeply understand "this specific passage is the best answer to this specific question."

### 12.1 Bi-Encoder vs Cross-Encoder

This is one of the most important concepts in modern retrieval systems.

**Bi-Encoder** (used for initial retrieval):

```
Query  →  Encoder  →  Vector
                              ↓
                          Similarity
                              ↑
Passage  →  Encoder  →  Vector
```

Fast, but the model never directly compares the query against the passage.

**Cross-Encoder** (used for reranking):

```
┌─────────────────────────────┐
│ Query + Passage together    │
│                             │
│ "How does JWT work?"        │
│ "The server validates..."   │
└──────────────┬──────────────┘
               ↓
        Cross-Encoder
               ↓
        Relevance Score
```

The model directly examines the query and passage together, producing better relevance judgments — but it's more expensive.

### 12.2 The Two-Stage Retrieval Pattern

This gives us a standard retrieval architecture:

```
                 Fast
                  │
                  ▼
          Candidate Retrieval
          (Semantic + BM25 + RRF)
                  │
                ~20
                  │
                  ▼
             Reranking
          (Cross-Encoder)
                  │
               Top 5
                  │
                  ▼
                 LLM
```

The principle:

> **Use cheap methods to find candidates, then expensive methods to select the best candidates.**

This is one of the most important patterns in retrieval system design.

### 12.3 Local vs API Reranking

Recall supports two reranking strategies:

|             | `local` (default)                        | `jina`                      |
| :---------- | :--------------------------------------- | :-------------------------- |
| **Privacy** | Fully local, no data leaves your machine | Query + chunks sent to Jina |
| **Latency** | No network, but uses CPU/GPU             | Network round-trip          |
| **Cost**    | Free                                     | API cost                    |
| **Setup**   | Model auto-downloads (~90MB)             | Requires API key            |

> **Implementation**: `app/lib/rerank.ts`

This is an example of an important AI Engineering decision:

> **There is rarely one universally "best" architecture. The correct choice depends on privacy, latency, infrastructure, cost, and scale.**

---

## 13. Phase 4 — Generation

After reranking, Recall keeps the best five passages. Those become the LLM's context.

### 13.1 Context Construction

The model receives retrieved passages as numbered context. Here's the actual prompt Recall sends to Gemini (from `app/lib/generate.ts`):

```
You are Recall, a retrieval-augmented assistant for a user's personal documents.

Answer the user's question using ONLY the context provided below.

Rules:
- Cite the supporting context inline using its marker number, e.g. [1] or [2].
- If a claim is supported by multiple sources, cite each one separately like [1][2]
  - never combine them into a single bracket like [1, 2].
- If the answer is not contained in the context, reply exactly:
  "I don't have that information in your documents."
- Do not use any knowledge outside the provided context.
- Be concise.

CONTEXT:
[[1]] (source: recall-faq.md, chunk 1)
Authentication uses JWT tokens. The server validates the token...

[[2]] (source: recall-setup.md, chunk 3)
To configure authentication, set the JWT_SECRET environment variable...

QUESTION:
How does authentication work?

ANSWER:
```

Notice the key design decisions:

| Decision                                           | Purpose                                       |
| :------------------------------------------------- | :-------------------------------------------- |
| **Numbered markers** `[[1]]`, `[[2]]`              | Let the model cite specific sources           |
| **"Answer using ONLY the context"**                | Constrains the model to retrieved information |
| **"Reply exactly: I don't have that information"** | Teaches abstention                            |
| **Source filenames included**                      | So the model knows where each chunk came from |

> **Retrieval determines the knowledge available to the model. Prompt construction determines how that knowledge is presented to the model.**

> **Implementation**: `app/lib/generate.ts`

### 13.2 Grounded Generation

Recall instructs the LLM to answer using the retrieved context — not its own internal knowledge:

```
Retrieved evidence  →  LLM  →  Grounded answer
```

rather than:

```
Question  →  LLM's internal knowledge  →  Possibly hallucinated answer
```

Grounding doesn't mathematically guarantee zero hallucinations, but it creates an architectural constraint: the model is instructed to base its response on retrieved evidence.

### 13.3 Citations

Recall requires the model to cite its claims:

```
The application uses BM25 alongside semantic retrieval [1].

The final five candidates are passed to the generation model [2].
```

Citations let users inspect the evidence behind an answer — especially important for knowledge management systems.

### 13.4 Hallucination and Abstention

One of the most important RAG behaviors is knowing when **not** to answer.

Recall instructs the model to say:

```
I don't have that information in your documents.
```

when the answer isn't present in the retrieved context.

This introduces the idea of **abstention**: an AI system should not always try to produce an answer. Sometimes the correct response is "I don't know based on the available data."

This is an important reliability principle in AI Engineering.

---

## 14. Streaming AI Responses

Recall doesn't wait for the entire answer before showing it. The `/api/search` endpoint uses **Server-Sent Events (SSE)** to stream the response:

```
sources    →  retrieved chunks (shown immediately)
token      →  answer text, one piece at a time
token      →  more text arrives
token      →  even more
done       →  finished
```

Instead of:

```
Question  →  [wait 5 seconds]  →  Complete answer
```

the user sees:

```
Question
   ↓
Sources appear
   ↓
Answer starts appearing
   ↓
More tokens arrive
   ↓
Complete
```

This is an example of an AI Engineering concern that is not purely an ML problem:

> **Perceived latency matters as much as raw model latency.**

---

## 15. End-to-End Request Flow

The complete system:

```
                    INGESTION
                    ─────────

Document
   │
   ▼
Parse
   │
   ▼
Chunk
   │
   ▼
Embedding Model
   │
   ├───────────────┐
   ▼               ▼
LanceDB          BM25
   │               │
   └───────┬───────┘
           │
        Indexed
           │
═══════════╪═════════════════════════════════
           │
           │            QUERY
           │            ─────
        User Query
           │
           ▼
      Query Embedding
           │
      ┌────┴─────┐
      ▼          ▼
 Vector Search  BM25
      │          │
      └────┬─────┘
           ▼
          RRF
           │
       ~20 candidates
           │
           ▼
      Cross-Encoder
           │
        Top 5
           │
           ▼
      Context Builder
           │
           ▼
         Gemini
           │
           ▼
    Grounded Answer
           │
           ▼
    Citations + SSE
           │
           ▼
           UI
```

---

## 16. Real Query Walkthrough

Let's trace what happens when a user asks "How does authentication work?" — step by step, with the actual files and functions involved.

### Step 1: The UI sends the query

The user types the question in `SearchBar.tsx` and clicks search. The `handleSearch` function in `app/page.tsx` sends a POST request:

```
POST /api/search
Body: { "query": "How does authentication work?" }
```

### Step 2: The API route receives the request

In `app/api/search/route.ts`:

```
1. Parse the request body → extract query string
2. Validate query is not empty
3. Embed the query → call embedTexts([query])
```

### Step 3: Query embedding

In `app/lib/embeddings.ts`:

```
"How does authentication work?"
   ↓
MiniLM-L6-v2 (runs locally)
   ↓
[0.12, -0.42, 0.81, 0.07, ...]  (384-dimensional vector)
```

### Step 4: Hybrid search

In `app/lib/search.ts`, two searches run in parallel:

**Vector search** (`app/lib/vectordb.ts`):

```
Query vector → LanceDB cosine similarity → top 50 chunks ranked by meaning
```

**BM25 search** (`app/lib/minisearch.ts`):

```
Query text → MiniSearch inverted index → top 50 chunks ranked by keyword match
```

### Step 5: Reciprocal Rank Fusion

In `app/lib/search.ts`, RRF combines both rankings:

```
Vector result: "JWT tokens are validated by..." (rank 1)
BM25 result:   "JWT tokens are validated by..." (rank 2)

RRF score = 1/(60+1) + 1/(60+2) = 0.03252
```

Documents appearing in both lists get boosted. Top 20 candidates are selected.

### Step 6: Reranking

In `app/lib/rerank.ts`, the cross-encoder re-scores each candidate:

```
Query: "How does authentication work?"
Passage: "JWT tokens are validated by the server..."
   ↓
Cross-Encoder (runs locally)
   ↓
Relevance score: 0.94  (high — passage directly answers the question)
```

Top 5 candidates are selected.

### Step 7: Prompt construction

In `app/lib/generate.ts`, the `buildPrompt` function creates the prompt:

```
You are Recall, a retrieval-augmented assistant...

CONTEXT:
[[1]] (source: recall-faq.md, chunk 1)
Authentication uses JWT tokens. The server validates...

[[2]] (source: recall-setup.md, chunk 3)
To configure authentication, set JWT_SECRET...

QUESTION:
How does authentication work?

ANSWER:
```

### Step 8: Gemini generates the answer

In `app/lib/generate.ts`, the prompt is sent to Gemini. The response streams back token by token:

```
Token 1: "Authentication"
Token 2: " uses"
Token 3: " JWT"
Token 4: " tokens"
...
```

### Step 9: SSE streaming to the UI

In `app/api/search/route.ts`, each token is sent as an SSE event:

```
event: sources
data: [{"id":"chunk-1","text":"Authentication uses JWT...","score":0.94}, ...]

event: token
data: Authentication

event: token
data:  uses

event: token
data:  JWT

...

event: done
```

### Step 10: The UI renders the answer

In `app/page.tsx`, the SSE events are parsed. Tokens are accumulated into the answer string. In `app/components/Answer.tsx`, the markdown is rendered. Citation markers like `[1]` become clickable links that scroll to the corresponding source chunk.

**Total flow:**

| Step | What happens                | Time      |
| :--- | :-------------------------- | :-------- |
| 1    | User types query            | —         |
| 2    | API receives request        | —         |
| 3    | Embedding generated locally | ~100ms    |
| 4-5  | Vector search + BM25 + RRF  | ~80ms     |
| 6    | Cross-encoder reranks       | ~200ms    |
| 7-8  | Gemini streams tokens       | ~500ms+   |
| 9-10 | SSE → UI renders            | Streaming |

---

## 17. Important Trade-offs

Building AI systems is mostly about managing trade-offs.

### Retrieval quality vs latency

More retrieval stages (Vector + BM25 + RRF + Reranker) improve quality, but every stage costs computation and time.

### Chunk size vs context

|               | Small chunks   | Large chunks   |
| :------------ | :------------- | :------------- |
| **Retrieval** | Precise        | Less precise   |
| **Context**   | Less per chunk | More per chunk |

### Local models vs API models

|              | Local                   | API                      |
| :----------- | :---------------------- | :----------------------- |
| **Privacy**  | Private                 | Data leaves your machine |
| **Cost**     | Free (compute)          | API fees                 |
| **Hardware** | Requires CPU/GPU        | No local requirements    |
| **Scaling**  | Limited by your machine | Scales easily            |

### More context vs less context

Sending many chunks to the LLM is tempting, but too much irrelevant context can:

- Increase token usage and cost
- Increase latency
- Distract the model
- Reduce answer quality

This is why Recall retrieves ~20 candidates, then reranks down to 5 before sending to the LLM.

---

## 18. RAG Failure Modes

A major part of AI Engineering is understanding how systems fail. Here are the six main failure modes, each with a concrete example.

### Failure 1 — Bad parsing

```
PDF  →  Incorrect text extraction  →  Bad chunks  →  Bad retrieval
```

**Example**: A PDF contains a table with authentication steps, but the PDF parser extracts the text in column order instead of row order:

```
Expected: "Step 1: Set JWT_SECRET Step 2: Enable HTTPS"
Got:      "Step 1: Enable HTTPS Step 2: Set JWT_SECRET"
```

The model cannot retrieve information that was never extracted correctly.

### Failure 2 — Bad chunking

```
Important information  →  Split incorrectly  →  Meaning lost  →  Poor retrieval
```

**Example**: A 180-word chunk splits right in the middle of a sentence:

```
Chunk A ends: "The authentication system uses"
Chunk B starts: "OAuth2 with Google and GitHub providers."
```

Neither chunk contains the complete idea. A query about "OAuth2 authentication" may not match either chunk well enough.

### Failure 3 — Embedding mismatch

The embedding model may not represent a particular domain or vocabulary well.

**Example**: A legal document uses the term "hereinafter referred to as the Lessor." The embedding model trained on general text may not understand that "Lessor" means "landlord" — causing poor retrieval for queries about the landlord.

### Failure 4 — Retrieval failure

The correct passage exists but never appears in the candidate set. This is a **critical failure** — if retrieval fails, the LLM has no correct context to work with.

**Example**: A user asks "What port does the server run on?" The correct chunk says "The server listens on port 3000." But because the query uses "port" while the chunk focuses on "server configuration," neither vector search nor BM25 ranks it in the top 20. The LLM has no context and guesses "8080."

### Failure 5 — Reranking failure

The correct passage is retrieved but ranked below irrelevant passages.

**Example**: RRF returns 20 candidates. The correct chunk is at rank 3. The cross-encoder reranker pushes it down to rank 8 because it focuses too heavily on keyword overlap rather than semantic relevance. The top 5 sent to the LLM don't contain the answer.

### Failure 6 — Generation failure

The correct context is available, but the LLM misinterprets it, ignores it, adds unsupported information, or produces incorrect citations.

**Example**: The context says "The API supports JSON and CSV formats." The LLM answers "The API supports JSON, CSV, and XML formats" — adding XML from its own training data despite the context never mentioning it.

> **RAG is not simply "add a vector database to an LLM."** It is an entire information retrieval and generation pipeline — and every stage is a potential failure point.

---

## 19. Evaluating a RAG System

A RAG system should not be evaluated with "it seems to work." You need measurable evidence.

Recall's evaluation harness (`scripts/evaluate.ts`) runs 46 queries against indexed documents and measures retrieval quality using three metrics. Let's walk through each one with a concrete example.

### Setting up the example

Suppose we run a query and the retrieval pipeline returns 5 chunks. Two of them are relevant (contain the answer):

| Rank | Chunk                                   | Relevant? |
| :--- | :-------------------------------------- | :-------- |
| 1    | "File uploads support PDF and Markdown" | No        |
| 2    | "Authentication uses JWT tokens"        | Yes       |
| 3    | "The server uses PostgreSQL"            | No        |
| 4    | "To reset your password, go to..."      | Yes       |
| 5    | "Error handling uses try/catch"         | No        |

Our relevance array is: `[0, 1, 0, 1, 0]`

### Recall@K

**Question**: "Did we find at least one good chunk within the top K results?"

Recall@K returns 1 if ANY of the top K chunks is relevant, 0 otherwise.

**Example with our data**:

| Metric       | Top K results     | Result                                   |
| :----------- | :---------------- | :--------------------------------------- |
| **Recall@1** | `[0]`             | **0** — no relevant chunk in top 1       |
| **Recall@3** | `[0, 1, 0]`       | **1** — relevant chunk at rank 2         |
| **Recall@5** | `[0, 1, 0, 1, 0]` | **1** — relevant chunks at ranks 2 and 4 |

**What this means for RAG**: If Recall@5 = 90%, then for 9 out of 10 questions, at least one relevant chunk appeared in the top 5 results sent to the LLM. The remaining 10% of questions will likely get bad answers because the LLM had no correct context.

---

### MRR (Mean Reciprocal Rank)

**Question**: "On average, how high was the first good chunk?"

MRR looks at where the FIRST relevant result appears and takes 1/rank:

| First relevant at rank | Score           |
| :--------------------- | :-------------- |
| Rank 1                 | 1/1 = **1.0**   |
| Rank 2                 | 1/2 = **0.5**   |
| Rank 3                 | 1/3 = **0.333** |
| Rank 4                 | 1/4 = **0.25**  |
| Rank 5                 | 1/5 = **0.2**   |
| No relevant result     | **0**           |

**Example with our data**:

The first relevant chunk is at rank 2.

```
MRR = 1/2 = 0.5
```

**Another example** — if the ranking were `[1, 0, 0, 1, 0]`:

```
First relevant chunk is at rank 1
MRR = 1/1 = 1.0  (perfect — the answer is right at the top)
```

**What this means**: MRR tells you how quickly users find answers. MRR = 0.8 means the first relevant chunk is usually in the top 1-2 results. MRR = 0.3 means it's typically buried around rank 3.

---

### nDCG@K (Normalized Discounted Cumulative Gain)

**Question**: "Are the good chunks ranked near the top?"

Unlike Recall@K (which just checks if a relevant chunk exists), nDCG rewards **correct ordering**. A ranking of `[1, 1, 0, 0, 0]` is better than `[0, 0, 1, 1, 0]` — even though both have 2 relevant chunks.

**How it works**:

Step 1: Calculate DCG (Discounted Cumulative Gain) of your ranking:

```
DCG = Σ relevance / log2(rank + 1)

For [0, 1, 0, 1, 0]:
  Rank 1: 0 / log2(2) = 0
  Rank 2: 1 / log2(3) = 0.631
  Rank 3: 0 / log2(4) = 0
  Rank 4: 1 / log2(5) = 0.431
  Rank 5: 0 / log2(6) = 0

DCG = 0 + 0.631 + 0 + 0.431 + 0 = 1.062
```

Step 2: Calculate IDCG (Ideal DCG) — the best possible ranking:

```
Best ranking: [1, 1, 0, 0, 0]  (all relevant chunks first)

IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
```

Step 3: Normalize:

```
nDCG = DCG / IDCG = 1.062 / 1.631 = 0.651
```

**What this means**:

| nDCG Score | Interpretation                                   |
| :--------- | :----------------------------------------------- |
| **1.0**    | Perfect ranking (all relevant chunks at the top) |
| **0.0**    | No relevant chunks found                         |
| **0.65**   | Good, but there's room to improve the ordering   |

---

### Putting it all together

Here's a complete example with 3 queries:

| Query       | Relevance Array   | Recall@5 | MRR      | nDCG@5   |
| :---------- | :---------------- | :------- | :------- | :------- |
| Query 1     | `[0, 1, 0, 1, 0]` | 1.0      | 0.50     | 0.65     |
| Query 2     | `[1, 0, 0, 0, 0]` | 1.0      | 1.00     | 1.00     |
| Query 3     | `[0, 0, 0, 0, 1]` | 1.0      | 0.20     | 0.26     |
| **Average** |                   | **1.0**  | **0.57** | **0.64** |

**Key insight**: All three queries have Recall@5 = 1.0 (a relevant chunk exists in the top 5). But MRR and nDCG reveal the difference — Query 3 finds the answer but buries it at rank 5, while Query 2 puts it right at the top.

### Why measure before and after reranking?

The benchmark reports every metric twice — once for RRF, once for reranked — so you can see whether the reranker actually helps:

```
RRF ranking  ──┐
               ├──►  compare  ──►  did the reranker help?
Reranked      ──┘
```

If reranking improves MRR from 0.5 to 0.8, the cross-encoder is earning its cost — it's moving relevant chunks closer to the top.

> **Being able to define, compute, and interpret Recall@K, MRR, and nDCG — and use them to compare two retrieval strategies — is a real, hireable AI Engineering skill.**

---

## 20. How to Study This Project

If you're new to AI, don't try to read everything at once. Follow this order:

1. **Run it.** Complete [Quick Start](#2-quick-start) and ask a few questions. Build intuition before theory.

2. **Read `app/lib/types.ts`.** Understand the shape of a `Chunk` — this is the single data structure the whole system passes around.

3. **Read in pipeline order:**
   - `parsers.ts` → how files become text
   - `chunker.ts` → how text becomes chunks
   - `embeddings.ts` → how chunks become vectors
   - `vectordb.ts` + `minisearch.ts` → how those are stored
   - `search.ts` → how queries find candidates (RRF)
   - `rerank.ts` → how candidates are narrowed (cross-encoder)
   - `generate.ts` → how chunks become an answer

4. **Run the benchmark.** `npm run eval` and watch the numbers change as you experiment.

5. **Break it on purpose.** The fastest way to understand a stage is to disable it and see what breaks.

> **Learning habit:** Predict what each file does before reading it, then compare your prediction with the actual code. The gap between them is where learning happens.

---

## 21. Experiment Lab

Concrete experiments. Each teaches a specific concept.

| #   | Experiment                     | How                                                                                       | What to observe                                                                                          |
| :-- | :----------------------------- | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| 1   | **Disable reranking**          | In `app/api/search/route.ts`, skip `rerankChunks` and use `searchHybrid` results directly | Re-run `npm run eval`. Does Recall@5 / MRR drop? This reveals the reranker's real value.                 |
| 2   | **Change chunk size**          | Edit `chunkSize` / `overlap` in `app/lib/chunker.ts` (try 100 vs 400 words)               | Re-index and re-evaluate. Smaller = more precise but may split ideas; larger = more context but noisier. |
| 3   | **Local vs API reranker**      | Set `RERANK_PROVIDER=jina` + `JINA_RERANKING_API_KEY`                                     | Compare latency and privacy trade-offs from [Section 12.3](#123-local-vs-api-reranking).                 |
| 4   | **Force abstention**           | Ask something not in any uploaded document                                                | Confirm the model says "I don't have that information" instead of guessing.                              |
| 5   | **Watch streaming**            | Open browser devtools Network tab during a search                                         | See `sources` arrive before the first `token` — the UX win from streaming.                               |
| 6   | **Bad retrieval → bad answer** | Upload a document, then ask about a topic it doesn't cover well                           | Trace where the pipeline fails using [Section 18](#18-rag-failure-modes).                                |

> After each experiment, re-run `npm run eval`. Using the benchmark as your feedback loop is the most important engineering habit this project teaches.

---

## 22. Final Mental Model

If you remember nothing else, remember this:

```
You do NOT need to train a model to build with AI.

You DO need to:
  - Turn documents into retrievable chunks
  - Retrieve the right chunks for a question
  - Rank them well
  - Ground an LLM in that context
  - Measure whether it actually worked
```

Recall is built entirely from **existing, off-the-shelf models**:

```
MiniLM (embeddings)  +  LanceDB (vectors)  +  MiniSearch (BM25)
   +  RRF (fusion)  +  Cross-Encoder (rerank)  +  Gemini (generation)
```

None of these were invented here. The engineering skill is in **wiring them into a system that is reliable, measurable, and honest about what it doesn't know.**

That is the essence of modern AI Engineering — and it is a skill you can list on a resume even if your background is "just" Full Stack JavaScript. The same strengths carry over: composing services, managing data flow, handling failures, and caring about latency and UX. AI Engineering is the next layer on top of the web stack you already know.

---

<div align="center">

**Project status:** Local-first RAG demo. Retrieval runs fully on-device; only final generation calls an external API.

</div>
