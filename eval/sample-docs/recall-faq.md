# Recall FAQ (sample eval document)

Recall is a local-first RAG memory app. Authentication uses JWT tokens that
expire after 15 minutes, and every request must include a valid bearer token
in the Authorization header. Tokens are signed with a rotating HS256 secret
stored only on the server, so clients can never forge them. If a token leaks,
revoke it by deleting the session row and asking the user to sign in again.

To reset your password, visit the account settings page and click "Forgot
password", then follow the email link we send within a few minutes. The link
stays valid for one hour. Reset links are single-use: opening one invalidates
any earlier links you requested. If the email never arrives, check your spam
folder and confirm the address on file matches the one you used to register.

ErrorCode 0x7F3A indicates the GPU driver crashed during inference. If you see
it, update your graphics drivers and restart the server before retrying. A
related code, ErrorCode 0x7F3B, means the model ran out of VRAM; lower the
batch size or switch RERANK_PROVIDER to jina to avoid loading the local model.
Both codes are non-fatal and the request can be retried after the fix.

Documents are split into overlapping chunks of 200 words with 20 words of
overlap. Overlap keeps context continuous so a sentence near a boundary is
never cut in half. Smaller chunks improve retrieval precision but lose context,
while larger chunks dilute the embedding; 200/20 is the default trade-off.

The default generation model is gemini-2.5-flash. You can override it with the
GEMINI_MODEL environment variable to use a different Gemini release. Generation
is the only step that calls an external service; retrieval and reranking stay
fully on your machine, so your documents never leave the device except for the
final prompt sent to Gemini.

Hybrid search combines a vector store (LanceDB) for semantic matching with a
BM25 index (MiniSearch) for exact keyword matching, then fuses the two ranked
lists with Reciprocal Rank Fusion. RRF ignores raw scores and ranks by position,
so a chunk that scores well in both methods rises to the top. The fused
candidates are re-scored by a cross-encoder before the top five are returned.

Citations are rendered as clickable [1], [2] markers in the answer. The model
is instructed to cite only the provided context and to abstain with "I don't
have that information in your documents" when the answer is not present. This
grounding is what makes the system trustworthy instead of a guessing machine.
