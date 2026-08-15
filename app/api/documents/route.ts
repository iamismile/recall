import { NextRequest, NextResponse } from "next/server";
import { getSources, deleteIndexByDocId } from "@/app/lib/minisearch";
import { deleteByDocId as deleteVectorByDocId } from "@/app/lib/vectordb";

/**
 * Lists indexed documents.
 *
 * Returns each document with its id, source file name, and chunk count.
 */
export async function GET() {
  try {
    // The BM25 index already tracks every chunk, so it is a
    // reliable source of truth for what has been indexed.
    const documents = await getSources();
    return NextResponse.json({ documents });
  } catch (error) {
    console.error("List documents error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

/**
 * Deletes a document (all of its chunks) from both search stores.
 *
 * The document id is passed as a query parameter, e.g. ?docId=abc123
 */
export async function DELETE(request: NextRequest) {
  try {
    const docId = request.nextUrl.searchParams.get("docId");
    if (!docId) {
      return NextResponse.json(
        { error: "docId query parameter is required" },
        { status: 400 },
      );
    }

    // Remove from both the vector store and the BM25 index.
    await Promise.all([deleteVectorByDocId(docId), deleteIndexByDocId(docId)]);

    return NextResponse.json({ success: true, docId });
  } catch (error) {
    console.error("Delete document error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
