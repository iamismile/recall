"use client";

import Answer from "@/app/components/Answer";
import DocumentManager from "@/app/components/DocumentManager";
import FileUpload from "@/app/components/FileUpload";
import Results from "@/app/components/Results";
import SearchBar from "@/app/components/SearchBar";
import { SearchResult } from "@/app/lib/types";
import { useState } from "react";

export default function Home() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<string | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [docsVersion, setDocsVersion] = useState(0);

  // Sends a search query to the API and processes the SSE response.
  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setMessage("");
    setAnswer(null);
    setAnswerError(null);
    setResults([]);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      // Non-streaming error (validation / server error) returns JSON.
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        setMessage(data.error || "Search failed");
        setIsLoading(false);
        return;
      }

      // Get a reader for the streaming response.
      const reader = res.body.getReader();

      // Decode binary stream data into text.
      const decoder = new TextDecoder();
      let buffer = "";
      let fullAnswer = "";

      // Read the SSE stream until the server closes it.
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Add the newly received data to the existing buffer.
        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by two newlines
        const events = buffer.split("\n\n");

        // The last item may be an incomplete event.
        // Keep it in the buffer and process it when more data arrives.
        buffer = events.pop() ?? "";

        // Process every complete SSE event.
        for (const evt of events) {
          let event = "";
          let data = "";

          // Parse the event name and data from the SSE message.
          for (const line of evt.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data = line.slice(5).trim();
          }

          if (event === "sources") {
            const sources = JSON.parse(data) as SearchResult[];
            setResults(sources);
            if (sources.length === 0) {
              setMessage("No results found. Upload some documents first.");
            }
          } else if (event === "token") {
            fullAnswer += JSON.parse(data) as string;
            setAnswer(fullAnswer);
          } else if (event === "error") {
            const errData = JSON.parse(data) as { message?: string };
            setAnswerError(errData.message ?? "Failed to generate answer");
          } else if (event === "done") {
            setIsLoading(false);
          }
        }
      }
      setIsLoading(false);
    } catch {
      setMessage("Network error");
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
      <Answer answer={answer} isLoading={isLoading} error={answerError} />
      <Results results={results} />
    </main>
  );
}
