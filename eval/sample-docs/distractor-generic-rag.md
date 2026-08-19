# Understanding Retrieval-Augmented Generation (background reading)

Retrieval-augmented generation, or RAG, is a technique for answering questions using a private corpus instead of the model's training data. A typical pipeline embeds a query and compares it against embedded passages from a document collection, then passes the closest passages to a language model as context.

Most RAG systems begin by splitting source documents into passages. Splitting can be done by character count, by sentence, or by token budget, and the choice affects both retrieval accuracy and the amount of context the generator sees. Overlapping windows are common so that a sentence near a boundary is not cut in half.

Embeddings are produced by a sentence transformer that maps text to a dense vector. Similar meanings land close together in the vector space, which lets a nearest-neighbor search find relevant passages even when the wording differs from the query. These vectors are usually normalized so that dot product and cosine similarity agree.

A keyword index such as BM25 complements embeddings by matching exact terms, which dense vectors sometimes miss. Hybrid systems fuse the two rankings, often with reciprocal rank fusion, to get the benefits of both semantic and lexical search.

A reranker, frequently a cross-encoder, can refine the fused candidates by scoring each query-passage pair directly. This is more accurate than comparing embeddings but slower, so it is applied only to a small set of top candidates.

Vector databases store the embeddings and provide fast similarity search. They may persist to disk or to an object store, and many support filtering by metadata. The choice of database influences latency and how easily the index can be shared across machines.
