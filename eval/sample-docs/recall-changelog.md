# Recall Changelog (sample eval document)

This document records notable releases of Recall.

Version 1.4 added cross-encoder reranking after hybrid retrieval. Before this release the pipeline stopped at reciprocal rank fusion; from 1.4 onward the fused candidates are passed to a cross-encoder that re-scores them and returns the top five. The change improved answer grounding on ambiguous questions.

Version 1.2 shipped streaming answers with inline citations. Earlier versions returned the full answer at once; 1.2 introduced Server-Sent Events so tokens arrive as they are generated, and citation markers in the answer link to the source chunks shown below it.

Version 1.0 launched the local vector and BM25 hybrid search. This was the first public release and established the core architecture: documents are chunked, embedded with a local model, stored in LanceDB, and mirrored into a MiniSearch BM25 index for keyword search.

Version 0.9 was a closed beta that supported only plain text and Markdown uploads. PDF support, the document manager, and the reranker were all added after the beta concluded.

Version 1.3 improved indexing throughput by batching embedding requests, which roughly halved the time to index a large PDF. It also added the air-gapped retrieval-only mode for privacy-focused deployments.
