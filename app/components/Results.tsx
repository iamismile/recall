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
            <span className="font-semibold text-blue-700">[{idx + 1}]</span>
            <span className="mx-1">·</span>
            <span className="font-medium text-gray-600">📄 {result.source}</span>
            <span className="mx-1">·</span>
            <span>chunk {result.chunkIndex + 1}</span>
            <span className="mx-1">·</span>
            <span>score {result.score.toFixed(3)}</span>
          </p>
          <p className="text-gray-800">{result.text}</p>
        </div>
      ))}
    </div>
  );
}
