import { chunkText } from "@/app/lib/chunker";
import { MAX_UPLOAD_SIZE } from "@/app/lib/config";
import { embedTexts } from "@/app/lib/embeddings";
import {
  addChunksToIndex,
  deleteIndexByDocId,
  getSources,
} from "@/app/lib/minisearch";
import { extractTextFromFile } from "@/app/lib/parsers";
import { Chunk } from "@/app/lib/types";
import {
  addChunks,
  deleteByDocId as deleteVectorByDocId,
} from "@/app/lib/vectordb";
import { randomUUID } from "crypto";
import fs from "fs/promises";
import { NextRequest, NextResponse } from "next/server";
import path from "path";

export async function POST(request: NextRequest) {
  let tempFilePath: string | null = null;

  try {
    // Get the uploaded file from the form data.
    const formData = await request.formData();
    const file = formData.get("file") as File | null;

    // Make sure a file was uploaded.
    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Check whether the file type is supported
    const fileExt = path.extname(file.name).toLowerCase();
    const allowedExts = [".txt", ".md", ".pdf"];
    if (!allowedExts.includes(fileExt)) {
      return NextResponse.json(
        { error: "Only .txt, .md, and .pdf files are supported" },
        { status: 400 },
      );
    }

    if (file.size > MAX_UPLOAD_SIZE) {
      return NextResponse.json(
        {
          error: `File too large. Maximum size is ${MAX_UPLOAD_SIZE / (1024 * 1024)} MB`,
        },
        { status: 400 },
      );
    }

    // Save the uploaded file temporarily. We need an actual file on disk
    // because our text extraction functions work with file paths.
    const tempDir = path.join(process.cwd(), "tmp");
    await fs.mkdir(tempDir, { recursive: true });

    // Generate a unique temporary file name.
    tempFilePath = path.join(tempDir, `${randomUUID()}${fileExt}`);

    // Convert the uploaded file into a Buffer and save it
    const buffer = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(tempFilePath, buffer);

    // Extract text from the uploaded file
    const text = await extractTextFromFile(tempFilePath, fileExt.slice(1));

    // Make sure the file actually contained readable text.
    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { error: "No text could be extracted from the file" },
        { status: 400 },
      );
    }

    // Split the document into smaller pieces.
    const chunks = chunkText(text);

    // One document id for the whole upload. Every chunk inherits it,
    // so the document can be deleted/updated as a single unit no matter
    // what the file is later renamed to.
    const docId = randomUUID();

    const chunkObjects: Chunk[] = chunks.map((chunkText, i) => ({
      id: randomUUID(),
      docId,
      text: chunkText,
      source: file.name,
      chunkIndex: i,
    }));

    // Convert every chunk into an embedding vector
    // These vectors will be used for semantic/vector search
    const vectors = await embedTexts(chunks);

    // If a document with the same file name was indexed before, find
    // its docId(s) and remove those first
    const existingDocIds = (await getSources())
      .filter((doc) => doc.source === file.name)
      .map((doc) => doc.docId);

    // Only delete if there are existing documents
    if (existingDocIds.length > 0) {
      await Promise.all([
        deleteVectorByDocId(existingDocIds[0]),
        deleteIndexByDocId(existingDocIds[0]),
      ]);
    }

    // Store the chunks in both search systems.
    // LanceDB: Vector/semantic search
    // MiniSearch: BM25/keyword search.
    await Promise.all([
      addChunks(chunkObjects, vectors),
      addChunksToIndex(chunkObjects),
    ]);

    return NextResponse.json({
      success: true,
      fileName: file.name,
      chunksIndexed: chunks.length,
    });
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  } finally {
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath);
      } catch (error) {
        console.error("Failed to remove temporary file:", error);
      }
    }
  }
}
