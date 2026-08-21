import {
  pipeline,
  env as transformersEnv,
  FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { GoogleGenAI } from "@google/genai";
import { env } from "./config";

// The Gemini API accepts up to 100 texts per request,
// so we use fewer round-trips there.
const LOCAL_BATCH_SIZE = 32;
const GEMINI_BATCH_SIZE = 100;

// Store downloaded Hugging Face models in a local directory.
// This prevents the model from being downloaded again on every run.
transformersEnv.cacheDir = "./.cache";

// Keep a single instance of the embedding model in memory.
// This avoids loading the model repeatedly for every embedding request.
let embedder: FeatureExtractionPipeline | null = null;

/**
 * Loads and returns the local embedding model.
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
async function getLocalEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embedder;
}

/**
 * Converts an array of text strings into numerical vector embeddings.
 *
 * Each text is transformed into a vector that represents its semantic
 * meaning. Texts are processed in fixed-size batches so memory usage
 * stays constant regardless of how many texts are passed in.
 *
 * Two providers are supported:
 *
 * 1. Local (default)
 *    - MiniLM runs on-device. No data leaves the machine.
 *    - Produces 384-dimensional vectors.
 *
 * 2. Gemini
 *    - Uses the Gemini Embedding API. Requires GEMINI_API_KEY.
 *    - Much faster on weak hardware (e.g. small cloud hosts).
 *    - Produces 3072-dimensional vectors.
 *
 * IMPORTANT: vectors from different providers (or dimensions) are not
 * compatible. If you change EMBED_PROVIDER, delete the data/ directory
 * and re-index all documents.
 */
export async function embedTexts(
  texts: string[],
  // Only used by the Gemini provider. Documents being indexed should
  // use "RETRIEVAL_DOCUMENT"; search queries should use "RETRIEVAL_QUERY".
  // The model optimizes the embedding for how it will be used.
  taskType?: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
): Promise<number[][]> {
  if (!texts.length) {
    throw new Error("Input texts array cannot be empty");
  }

  if (texts.some((text) => !text.trim())) {
    throw new Error("Input texts cannot contain empty strings");
  }

  return env.embedProvider === "gemini"
    ? embedViaGemini(texts, taskType)
    : embedLocally(texts);
}

/**
 * Embeds texts locally using the MiniLM pipeline.
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
async function embedLocally(texts: string[]): Promise<number[][]> {
  try {
    const model = await getLocalEmbedder();
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += LOCAL_BATCH_SIZE) {
      const batch = texts.slice(i, i + LOCAL_BATCH_SIZE);
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

/**
 * Scales a vector to unit length (L2 normalization).
 *
 * Gemini auto-normalizes only the full 3072-dimension output.
 * Truncated dimensions (768, 1536) must be normalized manually so
 * that similarity depends on vector direction, not magnitude.
 *
 * Normalizing an already-normalized vector is a no-op, so this is
 * safe to apply unconditionally.
 */
function l2Normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  return norm === 0 ? vector : vector.map((v) => v / norm);
}

/**
 * Embeds texts using the Gemini Embedding API.
 *
 * Useful when running on hardware too weak for local inference
 * The trade-off is that document text is sent to Google's API.
 *
 * Requires:
 *
 * GEMINI_API_KEY
 */
async function embedViaGemini(
  texts: string[],
  taskType?: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
): Promise<number[][]> {
  const apiKey = env.geminiApiKey;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set");
  }

  try {
    const ai = new GoogleGenAI({ apiKey });
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += GEMINI_BATCH_SIZE) {
      const batch = texts.slice(i, i + GEMINI_BATCH_SIZE);
      const response = await ai.models.embedContent({
        model: env.geminiEmbedModel,
        contents: batch,
        config: {
          taskType,
          outputDimensionality: env.geminiEmbedDimension,
        },
      });
      if (!response.embeddings?.length) {
        throw new Error("Gemini returned no embeddings");
      }
      vectors.push(
        ...response.embeddings.map((e) => l2Normalize(e.values as number[])),
      );
    }
    return vectors;
  } catch (error) {
    console.error("Failed to embed texts via Gemini:", error);

    if (error instanceof Error) {
      throw new Error(`Embedding failed: ${error.message}`);
    }

    throw new Error(`Embedding failed: ${String(error)}`);
  }
}
