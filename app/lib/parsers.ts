import fs from "fs/promises";
import { PDFParse } from "pdf-parse";

/**
 * Extracts readable text from a supported file.
 *
 * Supported file types:
 * - TXT
 * - Markdown
 * - PDF
 *
 * PDF files need a PDF parser because their contents are not
 * stored as simple plain text.
 *
 * TXT and Markdown files can be read directly as UTF-8 text.
 */
export async function extractTextFromFile(
  filePath: string,
  fileType: string,
): Promise<string> {
  if (fileType === "pdf") {
    // Read the PDF file as binary data
    const dataBuffer = await fs.readFile(filePath);

    // Create a PDF parser using the file data
    const parser = new PDFParse({
      data: dataBuffer,
    });

    // Extract the text from the PDF
    const result = await parser.getText();
    return result.text;
  }

  // TXT and Markdown files are plain text,
  // so we can read them directly as UTF-8
  return await fs.readFile(filePath, "utf-8");
}
