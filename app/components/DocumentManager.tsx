"use client";

import { useEffect, useState } from "react";

interface IndexedDocument {
  docId: string;
  source: string;
  chunks: number;
}

interface DocumentManagerProps {
  refreshKey: number;
  onDeleted?: () => void;
}

export default function DocumentManager({
  refreshKey,
  onDeleted,
}: DocumentManagerProps) {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const res = await fetch("/api/documents");
        const data = await res.json();
        if (!cancelled) {
          setDocuments(data.documents ?? []);
        }
      } catch {
        if (!cancelled) setError("Failed to load documents");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const handleDelete = async (doc: IndexedDocument) => {
    setDeleting(doc.docId);
    setError("");
    try {
      const res = await fetch(
        `/api/documents?docId=${encodeURIComponent(doc.docId)}`,
        { method: "DELETE" },
      );
      if (res.ok) {
        setDocuments((prev) => prev.filter((d) => d.docId !== doc.docId));
        onDeleted?.();
      } else {
        const data = await res.json();
        setError(data.error || "Delete failed");
      }
    } catch {
      setError("Network error");
    } finally {
      setDeleting(null);
    }
  };

  if (loading && documents.length === 0) {
    return <p className="text-sm text-gray-500">Loading documents…</p>;
  }

  if (documents.length === 0) {
    return <p className="text-sm text-gray-500">No documents indexed yet.</p>;
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-2">Indexed documents</h2>
      {error && <p className="text-sm text-red-500 mb-2">{error}</p>}
      <ul className="space-y-2">
        {documents.map((doc) => (
          <li
            key={doc.docId}
            className="flex items-center justify-between p-3 bg-white rounded-lg shadow"
          >
            <span className="text-sm text-gray-800">
              📄 {doc.source}{" "}
              <span className="text-gray-500">
                ({doc.chunks} chunk{doc.chunks === 1 ? "" : "s"})
              </span>
            </span>
            <button
              onClick={() => handleDelete(doc)}
              disabled={deleting === doc.docId}
              className="px-3 py-1 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
            >
              {deleting === doc.docId ? "Deleting…" : "Delete"}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
