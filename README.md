# Recall — Local-First RAG Memory

> **An AI Engineering learning project for understanding and building a production-style Retrieval-Augmented Generation (RAG) system.**

Recall is a local-first personal knowledge application where users upload `.txt`, `.md`, and `.pdf` documents and ask questions about them using natural language.

The interesting part of Recall is not only the application itself.

**Recall is designed as a practical AI Engineering laboratory.**

Every major component of the system represents an important concept used when building modern AI-powered applications:

- Document ingestion
- Text extraction
- Chunking
- Embeddings
- Vector databases
- Semantic search
- BM25 / lexical search
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Reranking
- Bi-encoders vs. cross-encoders
- Prompt construction
- Grounded generation
- Citations
- Hallucination mitigation
- Streaming with SSE
- Local inference
- External model APIs
- Retrieval / generation trade-offs
- Latency and performance considerations
- Evaluation of RAG systems

### What This Project Demonstrates

This project puts the following AI Engineering capabilities into practice:

- **RAG architecture** — designing ingestion and query pipelines end-to-end.
- **Retrieval** — embeddings, vector search, BM25, hybrid search, Reciprocal Rank Fusion, cross-encoder reranking.
- **Prompt engineering & grounding** — context construction, inline citations, hallucination abstention.
- **Evaluation** — measuring Recall@K, MRR, and nDCG before and after reranking.
- **Local-first AI** — running embedding and reranking models on-device with transformers.js.
- **Full-stack delivery** — Next.js app, Server-Sent Events streaming, REST APIs, configuration and environment handling.

The goal of this README is therefore twofold:

> **Learn the concept → understand why it exists → see the implementation → understand the trade-offs.**

---

# Table of Contents

## Setup & Familiarity

- [1. Quick Start](#1-quick-start)
- [2. Project Structure](#2-project-structure)
- [3. API Reference](#3-api-reference)
- [4. Configuration](#4-configuration)
- [5. Technology Stack](#5-technology-stack)
- [6. Evaluation Harness](#6-evaluation-harness)

## Learn The Concepts

- [1. What Is Recall?](#1-what-is-recall)
- [2. AI Engineering vs. AI Research](#2-ai-engineering-vs-ai-research)
- [3. What Is RAG?](#3-what-is-rag)
- [4. The Recall Architecture](#4-the-recall-architecture)
- [5. Phase 1 — Document Ingestion](#5-phase-1--document-ingestion)
- [6. Phase 2 — Retrieval](#6-phase-2--retrieval)
- [7. Phase 3 — Reranking](#7-phase-3--reranking)
- [8. Phase 4 — Generation](#8-phase-4--generation)
- [9. Streaming AI Responses](#9-streaming-ai-responses)
- [10. End-to-End Request Flow](#10-end-to-end-request-flow)
- [11. Important AI Engineering Trade-offs](#11-important-ai-engineering-trade-offs)
- [12. RAG Failure Modes](#12-rag-failure-modes)
- [13. Evaluating a RAG System](#13-evaluating-a-rag-system)
- [14. How to Study This Project](#14-how-to-study-this-project)
- [15. Experiment Lab](#15-experiment-lab)
- [16. Final Mental Model](#16-final-mental-model)

---

# Setup & Familiarity

## 1. Quick Start

### Prerequisites

- **Node.js 18+** (recommended: 20+)
- **npm** or pnpm
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- ~2GB free disk space (for model downloads)

### Install

```bash
git clone <repo-url>
cd recall
npm install
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
```

That's the only required key. Everything else runs locally.

### Start Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Upload a document (`.txt`, `.md`, or `.pdf`), wait for indexing to complete, then ask a question in natural language.

### First Run Notes

| Component           | First Run Behavior                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding model** | Downloads `all-MiniLM-L6-v2` (~90MB) to `.cache/` on first upload or query. Runs entirely on your machine — no API call.                                               |
| **Reranking model** | If `RERANK_PROVIDER=local` (default), downloads `ms-marco-MiniLM-L-6-v2` (~90MB) to `.cache/` on first search. Set `RERANK_PROVIDER=jina` to skip local model loading. |
| **Vector database** | LanceDB persists to `data/lancedb/`                                                                                                                                    |
| **Keyword index**   | MiniSearch persists to `data/minisearch.json`                                                                                                                          |

### Troubleshooting

**"GEMINI_API_KEY not found"**
Check that `.env` exists and contains `GEMINI_API_KEY=...`. Restart the dev server after editing.

**Model download fails**
The first query requires network access to HuggingFace model servers. Check your connection and retry. Models are cached in `.cache/` after successful download.

**Search returns "I don't have that information in your documents."**
This is expected behavior when no relevant chunk is found. Try uploading a document with content matching your question, or ask a different question.

**PDF parsing returns no text**
Some PDFs are image-based (scanned). Recall requires text-based PDFs. Try a `.txt` or `.md` file instead.

---

## 2. Project Structure

```
recall/
├── app/
│   ├── api/
│   │   ├── upload/route.ts        # POST /api/upload — ingestion endpoint
│   │   ├── search/route.ts        # POST /api/search — query endpoint with SSE
│   │   └── documents/route.ts     # GET/DELETE /api/documents — document management
│   │
│   ├── components/                # UI components
│   │   ├── FileUpload.tsx         # Drag-and-drop upload
│   │   ├── DocumentManager.tsx    # List / delete documents
│   │   ├── SearchBar.tsx          # Query input
│   │   ├── Results.tsx            # Retrieved chunks display
│   │   └── Answer.tsx             # Streamed answer with citations
│   │
│   └── lib/                       # Core AI pipeline
│       ├── parsers.ts             # Text extraction (.txt/.md/.pdf)
│       ├── chunker.ts             # Overlapping word-window chunking
│       ├── embeddings.ts          # On-device MiniLM embeddings
│       ├── vectordb.ts            # LanceDB vector store
│       ├── minisearch.ts          # MiniSearch BM25 index
│       ├── search.ts              # Hybrid search + RRF
│       ├── rerank.ts              # Cross-encoder reranking (local / jina)
│       ├── generate.ts            # Gemini grounded prompt + streaming
│       └── types.ts               # Shared TypeScript types
│
├── data/                          # Persisted indexes (auto-created)
│   ├── lancedb/                   # Vector database
│   └── minisearch.json            # BM25 index
│
├── eval/                          # Offline retrieval evaluation
│   ├── sample-docs/
│   │   ├── recall-faq.md          # Benchmark documents (indexed by the eval harness)
│   │   ├── recall-setup.md
│   │   └── recall-troubleshooting.md
│   └── qa.jsonl                   # Gold queries + expected snippets
│
├── scripts/
│   └── evaluate.ts                # Benchmark script (`npm run eval`)
│
├── .env.example                   # Environment variables template
├── .env                           # Your local configuration (not committed)
├── package.json
└── README.md
```

### The `lib/` Directory Maps to the AI Pipeline

```
parsers.ts
    ↓
chunker.ts
    ↓
embeddings.ts
    ↓
vectordb.ts + minisearch.ts
    ↓
search.ts
    ↓
rerank.ts
    ↓
generate.ts
```

This makes the project easy to study one concept at a time.

---

## 3. API Reference

### `POST /api/upload`

Uploads and indexes a document.

**Request**: `multipart/form-data` with a `file` field.

```bash
curl -X POST http://localhost:3000/api/upload \
  -F "file=@my-notes.md"
```

**Pipeline**:

```
Upload → Parse → Chunk → Embed → Store (LanceDB + MiniSearch)
```

**Response**:

```json
{
  "docId": "a1b2c3d4",
  "source": "my-notes.md",
  "chunkCount": 42
}
```

**Notes**:

- Documents are identified by a generated `docId`, not filename, so re-uploading the same filename upserts (replaces) the old document.
- Empty files or unsupported formats are rejected with a 400.
- Supported formats: `.txt`, `.md`, `.pdf` (text-based only).

---

### `POST /api/search`

Searches indexed documents and streams a grounded answer.

**Request**:

```json
{
  "query": "How does authentication work?"
}
```

**Response**: **Server-Sent Events (SSE)** stream.

| Event type | Payload   | Description                                          |
| ---------- | --------- | ---------------------------------------------------- |
| `sources`  | `Chunk[]` | Top retrieved chunks (sent first, before any tokens) |
| `token`    | `string`  | Answer text deltas as they're generated              |
| `error`    | `string`  | Generation failure message                           |
| `done`     | —         | Stream complete                                      |

**Pipeline**:

```
Query → Embed → Hybrid Search (RRF, top 20) → Rerank (cross-encoder, top 5)
      → Gemini → Grounded Answer + Citations
```

**Example SSE stream**:

```
event: sources
data: [{"id":"chunk-1","text":"...","docId":"a1b2c3","score":0.92}, ...]

event: token
data: The

event: token
data:  application

event: token
data:  uses

...

event: done
```

**Notes**:

- Retrieval (embeddings, vector search, BM25, RRF, reranking) happens **entirely on-device**.
- Only the final generation calls Gemini.
- Requires `GEMINI_API_KEY` to be set.
- If reranking fails (e.g., model load error), the system falls back to the RRF ranking.

---

### `GET /api/documents`

Lists all indexed documents.

**Response**:

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

**Response**:

```json
{
  "success": true
}
```

---

## 4. Configuration

All options live in `.env`. Copy `.env.example` to `.env` and adjust as needed.

| Variable                 | Required                       | Default                              | Purpose                                                   |
| ------------------------ | ------------------------------ | ------------------------------------ | --------------------------------------------------------- |
| `GEMINI_API_KEY`         | ✅ for generation              | —                                    | Google AI Studio API key. Retrieval works without it.     |
| `GEMINI_MODEL`           | No                             | `gemini-2.5-flash`                   | Generation model override.                                |
| `RERANK_PROVIDER`        | No                             | `local`                              | `local` = on-device cross-encoder; `jina` = API reranker. |
| `JINA_RERANKING_API_KEY` | Only if `RERANK_PROVIDER=jina` | —                                    | Jina API key.                                             |
| `JINA_RERANK_MODEL`      | No                             | `jina-reranker-v2-base-multilingual` | Jina model to use.                                        |

### Choosing a Reranker Provider

|              | `local` (default)                         | `jina`                               |
| ------------ | ----------------------------------------- | ------------------------------------ |
| **Model**    | `Xenova/ms-marco-MiniLM-L-6-v2`           | `jina-reranker-v2-base-multilingual` |
| **Runs on**  | Your machine                              | Jina's servers                       |
| **Privacy**  | Fully local, no data leaves               | Query + chunks sent to Jina          |
| **Latency**  | No network, but CPU/GPU load              | Network round-trip                   |
| **Setup**    | Model auto-downloads to `.cache/` (~90MB) | Requires API key                     |
| **Best for** | Privacy-first local use                   | Serverless / weak hardware           |

---

## 5. Technology Stack

| Area               | Technology                                   | Notes                                   |
| ------------------ | -------------------------------------------- | --------------------------------------- |
| **Framework**      | Next.js (App Router)                         | TypeScript                              |
| **Embeddings**     | `Xenova/all-MiniLM-L6-v2`                    | On-device via transformers.js           |
| **Vector DB**      | LanceDB                                      | Local, file-based, persisted in `data/` |
| **Keyword search** | MiniSearch                                   | BM25-style, persisted as JSON           |
| **Fusion**         | Reciprocal Rank Fusion (RRF)                 | `k=60`                                  |
| **Reranking**      | `ms-marco-MiniLM-L-6-v2` (local) or Jina API | Cross-encoder                           |
| **Generation**     | Google Gemini                                | `gemini-2.5-flash` via `@google/genai`  |
| **Streaming**      | Server-Sent Events (SSE)                     | Sources first, then tokens              |
| **Parsing**        | `pdf-parse` for PDFs                         | `.txt`/`.md` read directly              |

---

## 6. Evaluation Harness

Recall ships with an **offline retrieval evaluation** so you can measure whether the retrieval pipeline (hybrid search + RRF, then the cross-encoder reranker) actually surfaces the right chunk — without calling Gemini.

### Run It

```bash
npm run eval                 # re-index the sample doc, then evaluate
npm run eval -- --fresh      # wipe the index first for an isolated benchmark
npm run eval -- --no-ingest  # evaluate against already-indexed documents
```

### What It Measures

- **Dataset**: `eval/sample-docs/recall-faq.md` (indexed) + `eval/qa.jsonl` (gold queries with `expectedSnippet`).
- **Relevance**: A retrieved chunk is "relevant" if it contains the `expectedSnippet` string.
- **Pipeline**: Each query is embedded, searched with hybrid RRF (top 20), then reranked (top 5).
- **Metrics**: **Recall@k**, **MRR**, **nDCG@k** reported for both the raw RRF ranking and the reranked results, so you can see whether the reranker actually helps.

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

- **Retrieval failure** (wrong chunks found) → investigate chunking, embeddings, BM25, RRF, reranking.
- **Generation failure** (right chunks, wrong answer) → investigate prompt, grounding, LLM behavior.

The benchmark isolates the retrieval pipeline, making it cheap, reproducible, and fast — no Gemini calls, no generated text, just measurable retrieval quality.

See [Section 13: Evaluating a RAG System](#13-evaluating-a-rag-system) for the full tutorial on evaluation concepts.

---

# Learn The Concepts

## 1. What Is Recall?

Recall is a personal semantic memory application.

The user uploads documents:

```
PDF
Markdown
TXT
```

Recall processes those documents into searchable chunks.

When the user asks a question:

```
"How does authentication work in this project?"
```

Recall searches the user's documents, identifies the most relevant passages, reranks them, and gives those passages to an LLM.

The LLM then generates an answer based on the retrieved context.

The important design principle is:

> **The LLM should not be responsible for remembering the user's documents. Retrieval supplies the knowledge; the LLM performs reasoning and generation over that knowledge.**

---

## 2. AI Engineering vs. AI Research

Recall is primarily an **AI Engineering project**, not an AI research project.

AI research might ask:

> "Can we invent a better retrieval algorithm?"

AI Engineering asks:

> "How do we combine existing models and algorithms into a reliable system that solves a real problem?"

For example, Recall does not train its own embedding model.

Instead, it combines:

```
MiniLM
   +
LanceDB
   +
BM25
   +
RRF
   +
Cross-Encoder
   +
Gemini
```

into a complete AI application.

This is an important distinction.

Modern AI Engineering often involves:

```
Existing Models
       ↓
Data Processing
       ↓
Retrieval
       ↓
Ranking
       ↓
Prompting
       ↓
Generation
       ↓
Evaluation
       ↓
Production System
```

The engineering challenge is making the **whole pipeline work reliably**.

---

## 3. What Is RAG?

RAG stands for:

> **Retrieval-Augmented Generation**

A normal LLM works approximately like:

```
Question
   ↓
LLM
   ↓
Answer
```

The problem is that the LLM does not automatically have access to your private documents.

RAG adds a retrieval layer:

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

RAG therefore separates the problem into two major responsibilities:

### Retrieval

> Find the right information.

### Generation

> Use that information to produce a useful answer.

This separation is one of the most important concepts in modern AI applications.

---

## 4. The Recall Architecture

Recall uses two major phases.

### Offline / Ingestion Phase

This happens when a document is uploaded:

```
Upload
  ↓
Parse
  ↓
Chunk
  ↓
Generate Embeddings
  ↓
Build Indexes
  ├── Vector Index
  └── BM25 Index
```

### Online / Query Phase

This happens whenever the user asks a question:

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
          RRF
           │
       Top ~20
           │
           ▼
       Reranking
           │
        Top 5
           │
           ▼
      Context
           │
           ▼
         Gemini
           │
           ▼
    Grounded Answer
           │
           ▼
       Citations
```

The implementation maps closely to this architecture:

```
app/lib/
├── parsers.ts
├── chunker.ts
├── embeddings.ts
├── vectordb.ts
├── minisearch.ts
├── search.ts
├── rerank.ts
└── generate.ts
```

This makes the project useful as a learning resource because each AI concept has a corresponding implementation.

---

## 5. Phase 1 — Document Ingestion

> **Implementation checkpoint:** Before reading the code, predict what the ingestion pipeline must do. Then open the referenced file and compare your mental model with the implementation. This habit is intentional throughout the tutorial.

Before an AI system can retrieve information, the information must first be converted into a representation that the system can search.

Recall performs:

```
File
 ↓
Text
 ↓
Chunks
 ↓
Vectors
 ↓
Indexes
```

---

### 5.1 Parsing

### Concept

Different document formats represent information differently.

A PDF is not simply a string.

Markdown contains markup.

TXT contains plain text.

The first task is therefore:

> **Normalize different input formats into text.**

Recall supports:

```
.txt
.md
.pdf
```

Implementation:

```
app/lib/parsers.ts
```

The main responsibility is extracting usable text from the uploaded document.

---

### 5.2 Chunking

### Why can't we embed the entire document?

Imagine a 200-page PDF.

Creating one embedding for the entire PDF creates a representation of the document as a whole.

Now the user asks:

```
"What does the document say about JWT authentication?"
```

The embedding may represent hundreds of unrelated topics.

The relevant information becomes diluted.

Instead, Recall divides the document into smaller pieces:

```
Document
│
├── Chunk 1
├── Chunk 2
├── Chunk 3
├── ...
└── Chunk N
```

These chunks become the basic retrieval unit.

### Current implementation

```
chunkSize = 200 words
overlap   = 20 words
```

Implemented in:

```
app/lib/chunker.ts
```

### Why overlap?

Consider:

```
Chunk A:
"The authentication process begins by validating the user's..."

Chunk B:
"...JWT token before checking the user's permissions."
```

The important concept may cross the boundary.

Overlap preserves some surrounding context.

```
Chunk A
████████████████████
              ██████
              overlap

                    ████████████████████
                    Chunk B
```

### Important trade-off

There is no universally correct chunk size.

```
Small chunks
    ↓
More precise retrieval
    ↓
Less context

Large chunks
    ↓
More context
    ↓
Less precise retrieval
```

Chunking is therefore an **information retrieval design decision**, not merely text preprocessing.

---

### 5.3 Embeddings

An embedding converts text into a vector.

For example:

```
"How do I reset my password?"
```

might become something conceptually similar to:

```
[0.12, -0.42, 0.81, 0.07, ...]
```

The actual vector contains many dimensions.

The important idea is:

> **Semantic information is represented numerically so that mathematical similarity can be used for retrieval.**

Recall uses:

```
Xenova/all-MiniLM-L6-v2
```

through:

```
@huggingface/transformers
```

and runs the model locally.

Implementation:

```
app/lib/embeddings.ts
```

---

### Bi-Encoder

Recall's embedding model is a **bi-encoder**.

The query and document are encoded separately:

```
Query
  ↓
Encoder ──→ Query Vector

Document
  ↓
Encoder ──→ Document Vector
```

Then similarity is calculated between the vectors.

This allows document embeddings to be calculated once during ingestion.

That is extremely important for performance.

Instead of:

```
Every query
    ↓
Encode every document
```

we do:

```
During ingestion:
Document → Embedding → Store

During query:
Query → Embedding → Search stored vectors
```

This is the basic idea behind scalable semantic retrieval.

---

### 5.4 Vector Storage

Recall stores embeddings in:

```
LanceDB
```

Implementation:

```
app/lib/vectordb.ts
```

The vector database stores information such as:

```
chunk
embedding
docId
metadata
```

At query time:

```
Query
 ↓
Query embedding
 ↓
Vector similarity search
 ↓
Nearest chunks
```

This is called:

> **Approximate / vector similarity retrieval**

depending on the underlying indexing strategy.

---

### 5.5 Keyword Index

Recall also creates a second index using:

```
MiniSearch
```

Implementation:

```
app/lib/minisearch.ts
```

This index is used for lexical / BM25 search.

Why have two indexes?

Because semantic and lexical retrieval solve different problems.

---

## 6. Phase 2 — Retrieval

Retrieval is arguably the most important part of a RAG system.

A powerful LLM cannot compensate for completely irrelevant context.

A useful mental model is:

> **Bad retrieval → bad context → bad answer.**

Recall therefore uses multiple retrieval techniques.

---

### 6.1 Semantic Search

Semantic search asks:

> "Which chunks have similar meaning to this question?"

For example:

```
Query:
"How can I recover my account credentials?"
```

A document containing:

```
"To reset your password, navigate to the account settings..."
```

can still be retrieved even though the wording is different.

This is the main advantage of embeddings.

### Strengths

Semantic search is good at:

- Meaning
- Intent
- Synonyms
- Paraphrases
- Natural language questions

### Weaknesses

It can struggle with:

- Error codes
- IDs
- Product names
- Exact strings
- Rare technical tokens

For example:

```
ErrorCode 0x7F3A
```

is often better handled by lexical search.

---

### 6.2 BM25

BM25 is a classic information retrieval algorithm.

Recall uses MiniSearch for BM25-style lexical retrieval.

BM25 considers concepts such as:

### Term Frequency

How frequently does a query term occur?

### Inverse Document Frequency

How rare is the term across documents?

Rare terms are more informative.

For example:

```
"the"
```

is not very useful.

But:

```
"0x7F3A"
```

is highly specific.

### Document Length

A match inside a very long document should not automatically dominate a match inside a short, focused passage.

BM25 therefore provides strong lexical matching.

---

### 6.3 Why Hybrid Search?

Semantic search and BM25 have complementary strengths.

|             | Semantic       | BM25      |
| ----------- | -------------- | --------- |
| Meaning     | Excellent      | Weak      |
| Synonyms    | Excellent      | Weak      |
| Paraphrases | Excellent      | Weak      |
| Exact terms | Sometimes weak | Excellent |
| Error codes | Weak           | Excellent |
| Names / IDs | Sometimes weak | Excellent |

Therefore:

```
Semantic Search
       +
BM25
       ↓
Hybrid Retrieval
```

is generally more robust than relying on only one retrieval method.

---

### 6.4 Reciprocal Rank Fusion

Now we have two ranked lists.

Example:

```
Semantic:

A
B
C
D
E


BM25:

C
A
F
B
G
```

The problem is that their raw scores are not directly comparable.

For example:

```
Vector similarity:
0.82

BM25:
13.7
```

Adding those numbers does not make sense.

Instead, Recall uses:

> **Reciprocal Rank Fusion (RRF)**

The basic formula is:

```
RRF_score = 1 / (k + rank)
```

Recall uses:

```
k = 60
```

The important idea is that RRF cares about **rank position**, rather than the raw score.

A document appearing near the top of both lists receives a strong combined score.

```
Semantic       BM25
   │             │
   ▼             ▼
Ranked list   Ranked list
   │             │
   └──────┬──────┘
          ▼
         RRF
          │
          ▼
     Top ~20 candidates
```

RRF therefore acts as a relatively cheap **candidate-generation stage**.

---

## 7. Phase 3 — Reranking

After RRF we have approximately 20 candidates.

But 20 candidates are still too many to send directly to the LLM.

More importantly, RRF only knows:

> "This document ranked highly."

It does not deeply understand:

> "This specific passage is the best answer to this specific question."

That is where reranking comes in.

---

### 7.1 Bi-Encoder vs Cross-Encoder

This is one of the most important concepts in modern retrieval systems.

### Bi-Encoder

The query and passage are encoded separately.

```
Query ──────→ Encoder ─────→ Vector
                              │
                              ▼
                           Similarity
                              ▲
                              │
Passage ────→ Encoder ─────→ Vector
```

### Advantages

- Fast
- Embeddings can be precomputed
- Excellent for large-scale candidate retrieval

### Disadvantage

The model does not directly examine the query and passage together.

---

### Cross-Encoder

A cross-encoder receives both at the same time:

```
┌─────────────────────────────┐
│ Query + Passage             │
│                             │
│ "How does JWT work?"        │
│                             │
│ "The server validates..."   │
└──────────────┬──────────────┘
               ↓
        Cross-Encoder
               ↓
        Relevance Score
```

The model can directly compare the query against the passage.

This generally provides better relevance judgment.

But it is more expensive.

---

### 7.2 Why Reranking Exists

This gives us a standard retrieval architecture:

```
                     Fast
                      │
                      ▼
              Candidate Retrieval
                      │
            Semantic + BM25 + RRF
                      │
                    ~20
                      │
                      ▼
                 Reranking
                      │
                 Cross-Encoder
                      │
                     Top 5
                      │
                      ▼
                    LLM
```

The principle is:

> **Use cheap methods to find candidates, then expensive methods to select the best candidates.**

This is one of the most important patterns to understand when designing retrieval systems.

Recall implements this in:

```
app/lib/rerank.ts
```

---

### 7.3 Local vs API Reranking

Recall supports two reranking strategies.

### Local

```
RERANK_PROVIDER=local
```

Uses:

```
Xenova/ms-marco-MiniLM-L-6-v2
```

through transformers.js.

Advantages:

- Private
- No external request
- No API cost
- Works locally

Disadvantages:

- Uses local CPU / memory
- Model loading increases resource usage
- Potentially slower on weak hardware

### Jina API

```
RERANK_PROVIDER=jina
```

Advantages:

- No local reranker model
- Useful for serverless environments
- Less local memory usage

Disadvantages:

- Network latency
- API dependency
- API cost
- Data leaves the local environment

This is an example of an important AI Engineering decision:

> **There is rarely one universally "best" architecture.**

The correct choice depends on privacy, latency, infrastructure, cost, and scale.

---

## 8. Phase 4 — Generation

After reranking, Recall keeps the best five passages.

Those passages become the LLM's context.

```
Top 5 chunks
     ↓
Context construction
     ↓
Prompt
     ↓
Gemini
     ↓
Answer
```

Recall uses Gemini for final answer generation.

Implementation:

```
app/lib/generate.ts
```

---

### 8.1 Context Construction

The model receives the retrieved passages as numbered context:

```
[1] First relevant passage...

[2] Second relevant passage...

[3] Third relevant passage...
```

The prompt tells the model which information it is allowed to use.

This is an important RAG concept:

> **Retrieval determines the knowledge available to the model. Prompt construction determines how that knowledge is presented to the model.**

---

### 8.2 Grounded Generation

Recall instructs the LLM to answer using the retrieved context.

The desired flow is:

```
Retrieved evidence
       ↓
       LLM
       ↓
Grounded answer
```

rather than:

```
Question
   ↓
LLM's internal knowledge
   ↓
Possibly hallucinated answer
```

Grounding does not mathematically guarantee that hallucinations disappear.

Instead, it creates an architectural constraint:

> The model is instructed to base its response on retrieved evidence.

---

### 8.3 Citations

Recall requires the model to cite claims:

```
[1]
[2]
[3]
```

For example:

```
The application uses BM25 alongside semantic retrieval [1].

The final five candidates are passed to the generation model [2].
```

Citations make the system more useful because the user can inspect the evidence behind the answer.

This is especially important for knowledge-management systems.

---

### 8.4 Hallucination and Abstention

One of the most important RAG behaviors is knowing when **not** to answer.

Recall instructs the model to say:

```
I don't have that information in your documents.
```

when the answer is not present in the retrieved context.

This introduces the idea of:

> **Abstention**

An AI system should not always try to produce an answer.

Sometimes the correct answer is:

```
I don't know based on the available data.
```

This is an important reliability principle in AI Engineering.

---

## 9. Streaming AI Responses

Recall does not wait for the entire answer before sending it to the browser.

The `/api/search` endpoint uses:

> **Server-Sent Events (SSE)**

The response is streamed in stages:

```
sources
   ↓
token
   ↓
token
   ↓
token
   ↓
...
   ↓
done
```

This creates a much better user experience.

Instead of:

```
Question
   ↓
[wait 5 seconds]
   ↓
Complete answer
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

## 10. End-to-End Request Flow

The complete system can now be understood as:

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
           │
═══════════╪══════════════════════════════════
           │
           │            QUERY
           │            ─────
           │
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
           ├── Citations
           │
           ▼
          SSE
           │
           ▼
           UI
```

This is the core architecture to understand before reading the implementation.

---

## 11. Important AI Engineering Trade-offs

Building AI systems is mostly about managing trade-offs.

### Retrieval quality vs latency

More retrieval stages can improve quality:

```
Vector
+
BM25
+
RRF
+
Reranker
```

But every stage costs computation.

---

### Chunk size vs context

```
Small chunks
→ precise
→ less context

Large chunks
→ more context
→ less precise
```

---

### Local models vs API models

```
Local
→ privacy
→ no API cost
→ more hardware requirements

API
→ easy scaling
→ powerful infrastructure
→ network dependency
→ API cost
```

---

### More context vs less context

It is tempting to send many retrieved chunks to the LLM.

But more context does not automatically mean better answers.

Too much irrelevant context can:

- Increase token usage
- Increase latency
- Distract the model
- Reduce answer quality

This is why Recall performs:

```
~20 candidates
      ↓
reranking
      ↓
5 passages
      ↓
LLM
```

rather than sending everything retrieved.

---

## 12. RAG Failure Modes

A major part of AI Engineering is understanding how AI systems fail.

Recall can fail at several stages.

### Failure 1 — Bad parsing

```
PDF
 ↓
Incorrect text extraction
 ↓
Bad chunks
 ↓
Bad retrieval
```

The model cannot retrieve information that was never extracted correctly.

---

### Failure 2 — Bad chunking

```
Important information
       ↓
Split incorrectly
       ↓
Meaning lost
       ↓
Poor retrieval
```

---

### Failure 3 — Embedding mismatch

The embedding model may not represent a particular domain or vocabulary well.

---

### Failure 4 — Retrieval failure

The correct passage exists but never appears in the candidate set.

This is a critical failure.

If retrieval fails:

```
No correct context
      ↓
LLM cannot reliably answer
```

---

### Failure 5 — Reranking failure

The correct passage may be retrieved but ranked below irrelevant passages.

---

### Failure 6 — Generation failure

The correct context is available, but the LLM:

- Misinterprets it
- Ignores it
- Adds unsupported information
- Produces incorrect citations

Therefore:

> **RAG is not simply "add a vector database to an LLM."**

It is an entire information retrieval and generation pipeline.

---

## 13. Evaluating a RAG System

> **Implementation checkpoint:** The benchmark in `scripts/evaluate.ts` is not a separate toy system. It exercises the same retrieval components used by Recall, then measures the ranking before and after reranking. Read this section with the evaluation code open beside you.

A RAG system should not be evaluated with:

> "It seems to work."

A reliable RAG system needs measurable evidence that its retrieval and generation stages are doing their jobs.

For generation you can read the answer and judge it yourself. Retrieval is harder to eyeball: a system can _look_ like it works while silently returning the wrong chunk most of the time.

That is why Recall ships a retrieval benchmark in `scripts/evaluate.ts`. It answers one practical question:

> When a user asks a real question, does the pipeline actually surface the chunk that contains the answer?

### The three metrics, in plain English

**Recall@K** — _"Did we find at least one good chunk within the top K results?"_
This matters most for RAG: if the right chunk never enters the context, the LLM literally cannot answer. `Recall@5 = 90%` means that for 90% of questions, a relevant chunk appeared in the top 5.

**MRR (Mean Reciprocal Rank)** — _"On average, how high was the first good chunk?"_
A relevant chunk at rank 1 scores `1.0`; at rank 2 scores `0.5`; at rank 3 scores `~0.33`. MRR is the average across all questions. Higher means the model usually finds the answer quickly.

**nDCG@K** — _"Are the good chunks ranked near the top?"_
Unlike Recall@K, this rewards correct ordering. A ranking of `[good, good, bad]` beats `[bad, bad, good]` even though both eventually contain an answer. `1.0` is a perfect ranking; `0.0` is terrible.

### Why measure before and after reranking?

The benchmark reports every metric twice: once for the raw RRF ranking, once for the reranked ranking.

```
RRF ranking  ──┐
               ├─►  compare  ──►  did the reranker actually help?
Reranked      ──┘
```

If reranking improves `Recall@5` and `MRR`, the cross-encoder is earning its cost. If not, you have learned something concrete about your data — exactly the kind of evidence that separates _"I built a thing"_ from _"I engineered a system."_

> **Resume takeaway:** Being able to define, compute, and interpret **Recall@K, MRR, and nDCG** — and to use them to compare two retrieval strategies — is a real, hireable AI Engineering skill. You do not need a research background to demonstrate it.

---

## 14. How to Study This Project

If you are new to AI, do not try to read everything at once. Follow this order:

1. **Run it.** Complete [Quick Start](#1-quick-start) and ask a few questions. Build intuition before theory.
2. **Read `app/lib/types.ts`.** Understand the shape of a `Chunk` — this is the single data structure the whole system passes around.
3. **Read in pipeline order:**
   - `parsers.ts` → how files become text
   - `chunker.ts` → how text becomes chunks
   - `embeddings.ts` → how chunks become vectors
   - `vectordb.ts` + `minisearch.ts` → how those are stored
   - `search.ts` → how queries find candidates (RRF)
   - `rerank.ts` → how candidates are narrowed (cross-encoder)
   - `generate.ts` → how chunks become an answer
4. **Run the benchmark.** `npm run eval` and watch the numbers move as you change things (see [Experiment Lab](#15-experiment-lab)).
5. **Break it on purpose.** The fastest way to understand a stage is to disable it and see what breaks.

A good habit from [Section 2](#2-ai-engineering-vs-ai-research): predict what each file does, then read it and compare. The gap between your prediction and the code is where learning happens.

---

## 15. Experiment Lab

Concrete, low-risk experiments. Each one teaches a specific concept.

| #   | Experiment                     | How                                                                                                          | What to observe                                                                                                             |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Disable reranking**          | In `app/api/search/route.ts`, pass `FINAL_TOP_K` straight from `searchHybrid` without calling `rerankChunks` | Re-run `npm run eval`. Does `Recall@5` / `MRR` drop? This reveals the reranker's real value on your data.                   |
| 2   | **Change chunk size**          | Edit `chunkSize` / `overlap` in `app/lib/chunker.ts` (e.g. 100 vs 400)                                       | Re-index and re-evaluate. Smaller = more precise but may split ideas; larger = more context but noisier.                    |
| 3   | **Local vs API reranker**      | Set `RERANK_PROVIDER=jina` + `JINA_RERANKING_API_KEY`                                                        | Compare latency and privacy trade-offs from [7.3](#73-local-vs-api-reranking).                                              |
| 4   | **Force abstention**           | Ask something not in any document                                                                            | Confirm the model says _"I don't have that information"_ instead of guessing — see [8.4](#84-hallucination-and-abstention). |
| 5   | **Watch streaming**            | Open the browser devtools Network tab during a search                                                        | See `sources` arrive _before_ the first `token` — the UX win from [9](#9-streaming-ai-responses).                           |
| 6   | **Bad retrieval → bad answer** | Upload a document, then ask about a topic it does not cover well                                             | Trace where the pipeline fails using [12](#12-rag-failure-modes).                                                           |

After each experiment, re-run `npm run eval`. Treating the benchmark as your feedback loop is the single most important engineering habit this project teaches.

---

## 16. Final Mental Model

If you remember nothing else, remember this:

```
You do NOT need to train a model to build with AI.

You DO need to:
  - turn documents into retrievable chunks
  - retrieve the right chunks for a question
  - rank them well
  - ground an LLM in that context
  - measure whether it actually worked
```

Recall is built entirely from **existing, off-the-shelf models**:

```
MiniLM (embeddings)  +  LanceDB (vectors)  +  MiniSearch (BM25)
   +  RRF (fusion)  +  Cross-Encoder (rerank)  +  Gemini (generation)
```

None of these were invented here. The engineering skill is in **wiring them into a system that is reliable, measurable, and honest about what it does not know.**

That is the essence of modern AI Engineering — and it is a skill you can list on a resume even if your background is "just" Full Stack JavaScript. The same strengths carry over: composing services, managing data flow, handling failures, and caring about latency and UX. AI Engineering is the next layer on top of the web stack you already know.

---

> **Project status:** Local-first RAG demo. Retrieval runs fully on-device; only final generation calls an external API.
