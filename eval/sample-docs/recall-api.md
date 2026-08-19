# Recall API Reference (sample eval document)

Recall exposes a small HTTP API used by the web interface. All routes are under /api and expect JSON or multipart bodies depending on the operation.

POST /api/search streams Server-Sent Events containing sources, token, error, and done events. The client opens a POST request with a JSON body containing the query, then reads the event stream. The sources event carries the retrieved chunks, each token event carries a piece of the generated answer, and the done event signals completion. If generation fails, an error event is emitted but the sources are still delivered.

The upload route accepts a single multipart field named file. The server reads the uploaded bytes to a temporary file, extracts text based on the extension, chunks the text, embeds it, and writes the vectors to LanceDB and the terms to MiniSearch. Only .txt, .md, and .pdf are accepted; any other extension returns a 400 error. Re-uploading a file with the same name replaces the previous index entry.

DELETE /api/documents expects a docId query parameter and removes all of its chunks from both stores. The document manager UI calls this when you click delete. Because deletion targets the document id rather than the file name, renaming a file before re-uploading does not create orphaned entries.

GET /api/documents lists every indexed document with its chunk count. The response is a JSON object with a documents array; each entry has a docId, the source file name, and the number of chunks. The UI uses this to render the list of indexed files and to know how many chunks each one contributed.

A few implementation notes: the search route performs hybrid retrieval, reciprocal rank fusion, and optionally reranking before it begins streaming, so the first event may take a moment on a cold server. The upload route is synchronous and returns only after the document is fully indexed.
