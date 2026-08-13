import { NextRequest, NextResponse } from "next/server";
import { embedTexts } from "@/app/lib/embeddings";
import { searchSimilar } from "@/app/lib/vectordb";

export async function POST(request: NextRequest) {
  try {
    const { query } = await request.json();
    if (!query || query.trim().length === 0) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    // Embed the query
    const queryVector = (await embedTexts([query]))[0];

    // Search LanceDB
    const results = await searchSimilar(queryVector, 5);

    return NextResponse.json({ results });
  } catch (error) {
    console.error("Search error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
