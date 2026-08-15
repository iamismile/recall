import { GoogleGenAI } from "@google/genai";
import { SearchResult } from "./types";

// gemini-2.5-flash is fast, cheap, and GA. Override with GEMINI_MODEL
// if you want a different model (e.g. gemini-3-flash-preview).
const DEFAULT_MODEL = "gemini-2.5-flash";

/**
 * Generates a grounded answer from the retrieved contexts using Gemini.
 *
 * This is the "G" in RAG: we take the chunks that hybrid search
 * retrieved and ask the model to answer the question using ONLY
 * those chunks, citing them inline with [1], [2], ... markers.
 *
 * The context is numbered so the citations line up with the source
 * chunks we display to the user.
 */
export async function generateAnswer(
  query: string,
  contexts: SearchResult[],
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set");
  }

  const modelName = process.env.GEMINI_MODEL || DEFAULT_MODEL;
  const ai = new GoogleGenAI({ apiKey });

  // Numbered context block. The numbers double as citations the
  // model can reference inline (e.g. [1]).
  const contextText = contexts
    .map(
      (c, i) =>
        `[[${i + 1}]] (source: ${c.source}, chunk ${c.chunkIndex + 1})\n${c.text}`,
    )
    .join("\n\n");

  const prompt = `You are Recall, a retrieval-augmented assistant for a user's personal documents.

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

  const response = await ai.models.generateContent({
    model: modelName,
    contents: prompt,
  });

  return response.text ?? "";
}
