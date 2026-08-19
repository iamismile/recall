# Recall Troubleshooting (sample eval document)

If retrieval returns nothing, first confirm the document was actually indexed.
Open the document manager on the home page and check that the file appears in
the list with a non-zero chunk count. An empty index is the most common cause
of "no results" answers.

A slow first query is normal. The embedding model and the cross-encoder both
load from disk on their first use, and the embedding model in particular can
take several seconds to initialize on a cold machine. Subsequent queries are
fast because the models stay resident in memory.

If you see an out-of-memory error during indexing, index fewer or smaller
documents at a time. Embedding happens in a single batch per upload, so a very
large file can exhaust RAM before it is written to the vector store. Splitting
the source into several smaller files avoids the spike.

When reranking fails, the server logs the error and falls back to the RRF
ranking instead of aborting the request. You still get an answer; it is just
less precisely ordered. This fallback is intentional so search keeps working
even if the local model is unavailable.

Generation errors are streamed to the client as an error event while the
retrieved sources are still shown. If GEMINI_API_KEY is missing or invalid, the
answer step throws and the UI displays the failure message but keeps the
sources visible so you can inspect what was retrieved.

If the BM25 index and the vector store disagree about what is indexed, delete
the document and re-upload it. The upload endpoint removes every chunk for the
matching file name before re-adding, which resynchronizes both stores.
