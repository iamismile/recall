import { embedTexts } from "@/app/lib/embeddings";
import { streamAnswer } from "@/app/lib/generate";
import { searchHybrid } from "@/app/lib/search";
import { NextRequest, NextResponse } from "next/server";

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

    // Retrieve relevant document chunks using hybrid search.
    // Vector Search: Finds chunks with similar meaning.
    // BM25: Finds chunks containing relevant keywords.
    // RRF:  Combines the rankings from both search methods.
    // We retrieve the top 5 final results
    const results = await searchHybrid(query, queryVector, 5);

    // Create a Server-Sent Events (SSE) stream.
    // Client will receives:
    // 1. `sources` — retrieved document chunks.
    // 2. `token`   — generated answer text as it arrives.
    // 3. `error`   — generation error.
    // 4. `done`    — signals that the stream has finished.
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        // Helper function for sending an SSE event.
        const send = (event: string, data: unknown) =>
          controller.enqueue(
            encoder.encode(
              `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
            ),
          );

        // Send the retrieved sources before generating the answer.
        send("sources", results);

        // If no relevant documents were found, finish the stream.
        if (results.length === 0) {
          send("done", {});
          controller.close();
          return;
        }

        try {
          // Stream Gemini's response as it is generated.
          for await (const token of streamAnswer(query, results)) {
            send("token", token);
          }

          // Tell the client that generation has finished.
          send("done", {});
        } catch (err) {
          // Send generation errors to the client.
          send(
            "error",
            err instanceof Error ? err.message : "Failed to generate answer",
          );
        } finally {
          // Close the SSE connection.
          controller.close();
        }
      },
    });

    // `text/event-stream` tells the client that the response
    // contains Server-Sent Events instead of a normal JSON response.
    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Search error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
