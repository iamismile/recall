"use client";

import { useState } from "react";
import FileUpload from "@/app/components/FileUpload";
import SearchBar from "@/app/components/SearchBar";
import Results from "@/app/components/Results";
import DocumentManager from "@/app/components/DocumentManager";
import { SearchResult } from "@/app/lib/types";

export default function Home() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [docsVersion, setDocsVersion] = useState(0);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setMessage("");
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (res.ok) {
        setResults(data.results);
        if (data.results.length === 0) {
          setMessage("No results found. Upload some documents first.");
        }
      } else {
        setMessage(data.error || "Search failed");
      }
    } catch (err) {
      setMessage("Network error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="max-w-4xl mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Recall – Your Personal Semantic Memory
      </h1>
      <FileUpload onUploaded={() => setDocsVersion((v) => v + 1)} />
      <DocumentManager
        refreshKey={docsVersion}
        onDeleted={() => setDocsVersion((v) => v + 1)}
      />
      <SearchBar onSearch={handleSearch} isLoading={isLoading} />
      {message && <p className="text-center mt-4 text-red-500">{message}</p>}
      <Results results={results} />
    </main>
  );
}
