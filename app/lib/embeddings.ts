import {
  pipeline,
  env,
  FeatureExtractionPipeline,
} from "@huggingface/transformers";

// Store downloaded Hugging Face models in a local directory.
// This prevents the model from being downloaded again on every run.
env.cacheDir = "./.cache";

// Keep a single instance of the embedding model in memory.
// This avoids loading the model repeatedly for every embedding request.
let embedder: FeatureExtractionPipeline | null = null;

/**
 * Loads and returns the embedding model.
 *
 * The model is loaded only once and then reused for subsequent requests.
 * This is important because loading an ML model is an expensive operation.
 */
export async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embedder;
}

/**
 * Converts an array of text strings into numerical vector embeddings.
 *
 * Each text is transformed into a vector that represents its semantic meaning.
 *
 * pooling: "mean"
 * - Combines all word/token vectors into one vector for the whole text.
 *
 * normalize: true
 * - Makes all vectors the same length.
 * - This makes it easier to compare how similar two texts are.
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const model = await getEmbedder();
  const output = await model(texts, { pooling: "mean", normalize: true });
  return output.tolist() as unknown as number[][];
}
