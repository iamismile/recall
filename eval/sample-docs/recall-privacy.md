# Recall Privacy (sample eval document)

Recall is local-first: your documents are never uploaded to any third-party server. Everything you index stays in the data directory on your own machine, and the search and retrieval pipeline runs entirely locally. This design means your notes, PDFs, and Markdown files never leave the device except when you explicitly ask the assistant a question.

Only the final prompt is transmitted to the Gemini API, never your raw documents. When you submit a query, Recall builds a prompt that contains the retrieved context plus your question and sends that single request to Gemini. The model returns a streamed answer, and nothing about your documents is persisted by the provider beyond that request. If you are uncomfortable with even this, you can disable generation and use retrieval-only mode.

Model weights are cached in the .cache directory and stay on your machine. The embedding model and the reranker are downloaded once and then reused for every subsequent query, so no model data is re-fetched on each search. The cache is portable: copying the .cache folder to another machine avoids re-downloading the weights.

Telemetry is off by default and no usage data leaves the device. Recall does not phone home, does not collect analytics, and does not report which documents you have indexed. The only outbound network traffic is the optional call to Gemini when generating an answer, and the optional call to Jina when RERANK_PROVIDER is set to jina. Both are easy to disable by simply not setting the relevant keys.

For organizations that need stronger guarantees, Recall supports an air-gapped mode where the generation step is removed entirely. In that mode the app becomes a pure local retrieval tool: you still get ranked, cited sources, but no answer is generated and zero data leaves the network.
