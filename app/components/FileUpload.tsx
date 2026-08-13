"use client";

import { useState } from "react";

export default function FileUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(
          `✅ ${data.fileName} indexed (${data.chunksIndexed} chunks)`,
        );
      } else {
        setUploadStatus(`❌ ${data.error || "Upload failed"}`);
      }
    } catch (err) {
      setUploadStatus("❌ Network error");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  return (
    <div className="mb-6 p-4 border-2 border-dashed border-gray-300 rounded-lg text-center">
      <label className="cursor-pointer block">
        <span className="text-gray-600">
          {uploading
            ? "Uploading..."
            : "📁 Click to upload a file (.txt, .md, .pdf)"}
        </span>
        <input
          type="file"
          accept=".txt,.md,.pdf"
          onChange={handleFileChange}
          className="hidden"
          disabled={uploading}
        />
      </label>
      {uploadStatus && <p className="mt-2 text-sm">{uploadStatus}</p>}
    </div>
  );
}
