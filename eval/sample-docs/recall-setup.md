# Recall Setup (sample eval document)

Recall is installed like any Next.js app. Run npm install to fetch the
dependencies, then copy .env.example to .env and fill in your keys. The only
required key is GEMINI_API_KEY, which is used only for the final answer step.

Supported file types are plain text, Markdown, and PDF. Other formats are
rejected at upload time with a 400 response. PDF text is extracted with the
pdf-parse library, while text and Markdown are read directly as UTF-8.

The reranker runs on your machine by default. Set RERANK_PROVIDER=local to keep
the cross-encoder on-device, or RERANK_PROVIDER=jina to call the Jina Reranker
API instead. The Jina path needs JINA_RERANKING_API_KEY and avoids loading a
second model into process memory, which helps on small servers.

On-device models are downloaded once and cached in the .cache directory. The
embedding model is Xenova/all-MiniLM-L6-v2 and the reranker is
Xenova/ms-marco-MiniLM-L-6-v2. The first query is slow because both models load
then; later queries reuse the cached instances and are much faster.

Indexes persist on disk under the data directory. The vector store lives in
data/lancedb and the BM25 index lives in data/minisearch.json. Because both
persist, you can restart the server without re-indexing your documents.

To start developing, run npm run dev and open http://localhost:3000. Upload a
document from the home page, then type a question in the search bar to see the
retrieved sources and the streamed, cited answer.
