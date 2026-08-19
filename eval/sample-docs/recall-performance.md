# Recall Performance Tuning (sample eval document)

Recall's retrieval quality and speed depend on a few tuning knobs.

Lowering the chunk size improves retrieval precision at the cost of context. Smaller chunks mean each embedding represents a narrower piece of text, so semantic search can pinpoint the exact sentence that answers a question, but the model sees less surrounding context when generating the answer.

Batch embedding several chunks together uses GPU memory more efficiently. Rather than embedding one chunk at a time, the upload path sends the whole batch to the model in a single call, which amortizes overhead and keeps the GPU fed. Setting the batch size too high is what causes out-of-memory errors on small machines.

Increasing RRF_K makes the fused ranking depend more on high-rank agreement. RRF adds 1/(k+rank) for each method; a larger k flattens the contribution of rank, so a chunk must do well in both vector and keyword search to rise. A smaller k lets a strong showing in just one method dominate.

If the first search feels slow, remember that the models are loading; later searches are fast. You can pre-warm the models by issuing a dummy query at server start.

For very large documents, prefer splitting the source into several smaller files rather than one giant file, because embedding happens in a single batch per upload.
