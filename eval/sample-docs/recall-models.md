# Recall Models (sample eval document)

Recall relies on two small Transformer models that run on your machine.

The embedding model Xenova/all-MiniLM-L6-v2 outputs 384-dimensional vectors. Each chunk of text is passed through the model, the token representations are mean-pooled, and the result is L2-normalized to produce a single fixed-length vector that captures the chunk's meaning.

The reranker Xenova/ms-marco-MiniLM-L-6-v2 is a cross-encoder trained on MS MARCO. Unlike the bi-encoder embedding model, the cross-encoder receives the query and a candidate document together and outputs a single relevance score. It is more accurate than cosine similarity but slower, which is why it is applied only to the handful of candidates produced by hybrid search.

Vectors are L2-normalized so cosine similarity equals the dot product. Because every embedding has unit length, ranking by dot product is equivalent to ranking by cosine similarity, and the vector database can use a simple inner-product scan instead of a more expensive distance computation.

The embedding model has a maximum input length of 256 tokens, so chunks longer than that are truncated. The default chunk size of 180 words is chosen to stay safely under that limit for typical English text, though very dense prose can still approach the boundary.

Both models are loaded lazily on first use and then cached in memory. The first query pays the loading cost; later queries reuse the already-instantiated pipelines and are significantly faster.
