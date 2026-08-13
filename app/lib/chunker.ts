export function chunkText(
  text: string,
  chunkSize = 200,
  overlap = 20,
): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + chunkSize, words.length);
    const chunk = words.slice(start, end).join(" ");
    chunks.push(chunk);
    if (end === words.length) break;
    start = end - overlap;
  }
  return chunks;
}
