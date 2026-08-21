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
 * Model:
 * - Xenova/all-MiniLM-L6-v2
 * - A lightweight Transformer-based sentence embedding model.
 * - Produces a 384-dimensional vector for each input text.
 * - Designed to capture the semantic meaning of text rather than
 *   simply matching exact words.
 *
 * Limitations:
 * - Maximum input length: 256 tokens
 * - In typical English text, 256 tokens is often around 180–200 words,
 *   but this varies depending on the text.
 * - Text longer than this will be truncated silently
 *
 * The model is loaded only once and then reused for subsequent requests.
 * Loading an ML model is expensive, so keeping the instance in memory
 * avoids paying that cost for every embedding request.
 */
export async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embedder;
}

// Number of texts embedded per model forward pass.
// The pipeline pads the whole input array into a single tensor,
// so embedding a large document in one call spikes memory
// Small batches keep peak memory flat.
const EMBED_BATCH_SIZE = 16;

/**
 * Converts an array of text strings into numerical vector embeddings.
 *
 * Each text is transformed into a 384-dimensional vector that represents
 * its semantic meaning.
 *
 * Texts are processed in fixed-size batches so memory usage stays
 * constant regardless of how many texts are passed in.
 *
 * `pooling: "mean"`
 * - The Transformer produces a vector for each token.
 * - Mean pooling combines those token vectors into one vector
 *   representing the entire text.
 *
 * `normalize: true`
 * - Changes each vector so that its length becomes 1 applying L2 normalization (Euclidean norm)
 * - It does not change the direction of the vector.
 * - This makes vectors easier to compare when searching for
 *   texts with similar meanings.
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (!texts.length) {
    throw new Error("Input texts array cannot be empty");
  }

  if (texts.some((text) => !text.trim())) {
    throw new Error("Input texts cannot contain empty strings");
  }

  try {
    const model = await getEmbedder();
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
      const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
      const output = await model(batch, { pooling: "mean", normalize: true });
      vectors.push(...(output.tolist() as unknown as number[][]));
    }
    return vectors;
  } catch (error) {
    console.error("Failed to embed texts:", error);

    // Type guard to check if error has message property
    if (error instanceof Error) {
      throw new Error(`Embedding failed: ${error.message}`);
    }

    // Handle non-Error objects
    throw new Error(`Embedding failed: ${String(error)}`);
  }
}
