import fs from "fs/promises";
import { PDFParse } from "pdf-parse";

export async function extractTextFromFile(
  filePath: string,
  fileType: string,
): Promise<string> {
  if (fileType === "pdf") {
    const dataBuffer = await fs.readFile(filePath);
    const parser = new PDFParse({
      data: dataBuffer,
    });
    const result = await parser.getText();
    return result.text;
  } else {
    // Assume text-based files (txt, md, etc.)
    return await fs.readFile(filePath, "utf-8");
  }
}
