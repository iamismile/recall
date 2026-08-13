import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";
import { extractTextFromFile } from "@/app/lib/parsers";
import { chunkText } from "@/app/lib/chunker";
import { embedTexts } from "@/app/lib/embeddings";
import { addChunks } from "@/app/lib/vectordb";
import { Chunk } from "@/app/lib/types";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Save file temporarily
    const fileExt = path.extname(file.name).toLowerCase();
    const allowedExts = [".txt", ".md", ".pdf"];
    if (!allowedExts.includes(fileExt)) {
      return NextResponse.json(
        { error: "Only .txt, .md, and .pdf files are supported" },
        { status: 400 },
      );
    }

    const tempDir = path.join(process.cwd(), "tmp");
    await fs.mkdir(tempDir, { recursive: true });
    const tempFilePath = path.join(tempDir, `${randomUUID()}${fileExt}`);
    const buffer = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(tempFilePath, buffer);

    // Extract text
    const text = await extractTextFromFile(tempFilePath, fileExt.slice(1));
    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { error: "No text could be extracted from the file" },
        { status: 400 },
      );
    }

    // Chunk and embed
    const chunks = chunkText(text);
    const chunkObjects: Chunk[] = chunks.map((chunkText, i) => ({
      id: randomUUID(),
      text: chunkText,
      source: file.name,
      chunkIndex: i,
    }));

    const vectors = await embedTexts(chunks);

    // Store in LanceDB
    await addChunks(chunkObjects, vectors);

    // Clean up temp file
    await fs.unlink(tempFilePath);

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
  }
}
