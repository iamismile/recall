"use client";

import { SearchResult } from "@/app/lib/types";

interface ResultsProps {
  results: SearchResult[];
}

export default function Results({ results }: ResultsProps) {
  if (results.length === 0) return null;

  return (
    <div className="space-y-4">
      {results.map((result, idx) => (
        <div key={idx} className="p-4 bg-white rounded-lg shadow">
          <p className="text-sm text-gray-500 mb-2">
            Source: {result.source} (score: {result.score.toFixed(3)})
          </p>
          <p className="text-gray-800">{result.text}</p>
        </div>
      ))}
    </div>
  );
}
