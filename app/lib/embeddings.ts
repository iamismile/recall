import {
  pipeline,
  env,
  FeatureExtractionPipeline,
} from "@huggingface/transformers";

// Set model cache directory (optional, but keeps it local)
env.cacheDir = "./.cache";

let embedder: FeatureExtractionPipeline | null = null;

export async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embedder;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  const model = await getEmbedder();
  const output = await model(texts, { pooling: "mean", normalize: true });
  // `output.data` is a *flat* 1D array holding `batch * dim` values, so it cannot
  // be sliced per text. `output.tolist()` reshapes it according to `output.dims`
  // (e.g. [batch, 384]) into a proper number[][] with one vector per input text.
  return output.tolist() as unknown as number[][];
}
