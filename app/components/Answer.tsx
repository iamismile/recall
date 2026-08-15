"use client";

interface AnswerProps {
  answer: string | null;
  isLoading: boolean;
  error: string | null;
}

export default function Answer({ answer, isLoading, error }: AnswerProps) {
  if (isLoading) {
    return (
      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-blue-700">Generating answer…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-yellow-800 text-sm">
          Answer could not be generated ({error}), but matching sources are
          shown below.
        </p>
      </div>
    );
  }

  if (!answer) return null;

  return (
    <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <h2 className="text-sm font-semibold text-blue-800 mb-2">
        Answer
      </h2>
      <p className="text-gray-800 whitespace-pre-wrap">{answer}</p>
    </div>
  );
}
