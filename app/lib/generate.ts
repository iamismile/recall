import { GoogleGenAI } from "@google/genai";
import { SearchResult } from "./types";

// Gemini model used to generate the final answer.
// Can change it using the GEMINI_MODEL environment variable.
const DEFAULT_MODEL = "gemini-2.5-flash";

/**
 * Creates the prompt that will be sent to Gemini.
 *
 * This is where we connect retrieval with generation.
 *
 * The search system has already found relevant document chunks.
 * We give those chunks to Gemini as context and ask it to answer
 * the user's question using only that information.
 *
 * Each chunk gets a number such as [1], [2], etc.
 * Gemini can use these numbers when citing its answer.
 */
function buildPrompt(query: string, contexts: SearchResult[]): string {
  // Convert the retrieved chunks into a numbered context.
  // Example:
  // [1] source: notes.md
  // Authentication uses JWT tokens...
  const contextText = contexts
    .map(
      (c, i) =>
        `[[${i + 1}]] (source: ${c.source}, chunk ${c.chunkIndex + 1})\n${c.text}`,
    )
    .join("\n\n");

  // Tells Gemini how it should use the retrieved information.
  return `You are Recall, a retrieval-augmented assistant for a user's personal documents.

Answer the user's question using ONLY the context provided below.

Rules:
- Cite the supporting context inline using its marker number, e.g. [1] or [2].
- If the answer is not contained in the context, reply exactly: "I don't have that information in your documents."
- Do not use any knowledge outside the provided context.
- Be concise.

CONTEXT:
${contextText}

QUESTION:
${query}

ANSWER:`;
}

/**
 * Generates an answer using the retrieved document chunks
 *
 * Yields text deltas as they arrive so the UI can render the answer
 * token-by-token. Throws if the API key is missing.
 */
export async function* streamAnswer(
  query: string,
  contexts: SearchResult[],
): AsyncGenerator<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set");
  }

  const modelName = process.env.GEMINI_MODEL || DEFAULT_MODEL;
  const ai = new GoogleGenAI({ apiKey });

  // Build the prompt using the user's question and retrieved chunks.
  const prompt = buildPrompt(query, contexts);

  // Gemini generates the answer as a stream.
  const stream = await ai.models.generateContentStream({
    model: modelName,
    contents: prompt,
  });

  // Read each piece of the response as it arrives.
  // `yield` sends each piece to the caller immediately.
  for await (const chunk of stream) {
    const text = chunk.text;
    if (text) yield text;
  }
}
