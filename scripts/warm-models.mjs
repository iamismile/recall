// Pre-downloads ML models into .cache so the Docker image ships with them.
// This avoids slow first-request downloads on hosts with ephemeral disks.
//
// Set WARM_RERANKER=true to also cache the local cross-encoder reranker.
// Skip it when deploying with RERANK_PROVIDER=jina to keep the image small.
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from "@huggingface/transformers";

env.cacheDir = "./.cache";

console.log("Downloading embedding model: Xenova/all-MiniLM-L6-v2");
await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

if (process.env.WARM_RERANKER === "true") {
  console.log("Downloading reranker model: Xenova/ms-marco-MiniLM-L-6-v2");
  await AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
  await AutoModelForSequenceClassification.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2");
} else {
  console.log("Skipping reranker model (WARM_RERANKER != true)");
}

console.log("Model caching done.");
