/**
 * Splits text into smaller overlapping chunks.
 *
 * Why do we chunk?
 *
 * Large documents should not be stored or searched as one
 * large piece of text. Smaller chunks allow search to find
 * the specific part of a document that matches the query.
 *
 * `chunkSize`
 * - Maximum number of words in each chunk.
 * - Default: 180 words.
 *
 * `overlap`
 * - Number of words shared between consecutive chunks.
 * - Default: 20 words.
 * - Helps preserve context between chunks.
 *
 * Example:
 *
 * chunkSize = 180
 * overlap = 20
 *
 * Chunk 1 → words 0-179
 * Chunk 2 → words 160-339
 * Chunk 3 → words 320-499
 */
export function chunkText(
  text: string,
  chunkSize = 180,
  overlap = 20,
): string[] {
  if (!text.trim()) {
    return [];
  }

  if (chunkSize <= 0) {
    throw new Error("chunkSize must be greater than 0");
  }

  if (overlap < 0 || overlap >= chunkSize) {
    throw new Error(
      "overlap must be greater than or equal to 0 and less than chunkSize",
    );
  }

  // Split the document into words.
  const words = text.trim().split(/\s+/);

  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + chunkSize, words.length);

    // Create a chunk from the current range of words.
    const chunk = words.slice(start, end).join(" ");

    chunks.push(chunk);

    // Stop when we reach the end of the document.
    if (end === words.length) {
      break;
    }

    // Move forward while keeping the overlap.
    start = end - overlap;
  }

  return chunks;
}
