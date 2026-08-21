// Pre-downloads ML models into .cache so the Docker image ships with them.
// This avoids slow first-request downloads on hosts with ephemeral disks.
//
// Both downloads are opt-in via env vars so production images stay slim:
// - WARM_EMBEDDINGS=true  caches the local MiniLM embedding model
//   (not needed when using EMBED_PROVIDER=gemini)
// - WARM_RERANKER=true    caches the local cross-encoder reranker
//   (not needed when using RERANK_PROVIDER=jina)
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from "@huggingface/transformers";

env.cacheDir = "./.cache";

if (process.env.WARM_EMBEDDINGS === "true") {
  console.log("Downloading embedding model: Xenova/all-MiniLM-L6-v2");
  await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
} else {
  console.log("Skipping embedding model (WARM_EMBEDDINGS != true)");
}

if (process.env.WARM_RERANKER === "true") {
  console.log("Downloading reranker model: Xenova/ms-marco-MiniLM-L-6-v2");
  await AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
  await AutoModelForSequenceClassification.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
} else {
  console.log("Skipping reranker model (WARM_RERANKER != true)");
}

console.log("Model caching done.");
