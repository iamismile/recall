import { NextRequest, NextResponse } from "next/server";
import { embedTexts } from "@/app/lib/embeddings";
import { searchHybrid } from "@/app/lib/search";

export async function POST(request: NextRequest) {
  try {
    // Get the user's search query from the request body
    const { query } = await request.json();

    // Make sure the user actually provided a query.
    if (!query || query.trim().length === 0) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    // Convert the user's query into an embedding vector
    const queryVector = (await embedTexts([query]))[0];

    /**
     * Perform hybrid search.
     *
     * Vector Search:
     * - Finds documents with similar meaning.
     *
     * BM25:
     * - Finds documents with matching keywords.
     *
     * RRF:
     * - Combines the rankings from both search methods.
     *
     * Return the top 5 results.
     */
    const results = await searchHybrid(query, queryVector, 5);

    return NextResponse.json({ results });
  } catch (error) {
    console.error("Search error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
