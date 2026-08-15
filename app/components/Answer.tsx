"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AnswerProps {
  answer: string | null;
  isLoading: boolean;
  error: string | null;
}

// Turns plain "[1]", "[2]" citation markers in the model's answer into
// markdown links ([1](#ref-1)) so they become clickable. The matching
// source chunk in the Results list carries id="ref-1", etc.
function linkifyCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, (_m, n) => `[${n}](#ref-${n})`);
}

// Scrolls to the referenced source chunk and briefly highlights it.
function scrollToSource(refId: string) {
  const el = document.getElementById(refId);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("ring-2", "ring-blue-400", "bg-blue-50");
  window.setTimeout(() => {
    el.classList.remove("ring-2", "ring-blue-400", "bg-blue-50");
  }, 1600);
}

// Minimal Tailwind styling for the rendered markdown so we don't need
// an extra typography plugin. Each component only forwards the props it
// needs (className/children/href) — we intentionally avoid spreading the
// internal `node` prop onto DOM elements.
const mdComponents = {
  h1: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className="text-xl font-bold mb-2">{children}</h1>
  ),
  h2: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className="text-lg font-semibold mb-2">{children}</h2>
  ),
  h3: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className="text-base font-semibold mb-1">{children}</h3>
  ),
  p: ({ children }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="mb-2 leading-relaxed">{children}</p>
  ),
  ul: ({ children }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className="list-disc pl-5 mb-2">{children}</ul>
  ),
  ol: ({ children }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="list-decimal pl-5 mb-2">{children}</ol>
  ),
  li: ({ children }: React.HTMLAttributes<HTMLLIElement>) => (
    <li className="mb-1">{children}</li>
  ),
  a: ({ href, children }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
    // Citation links ([1] -> #ref-1): scroll to the source chunk and
    // render a clean [n] badge instead of the raw link text.
    if (href && href.startsWith("#ref-")) {
      const n = href.slice("#ref-".length);
      return (
        <a
          href={href}
          onClick={(e) => {
            e.preventDefault();
            scrollToSource(href.slice(1));
          }}
          title={`Jump to source ${n}`}
          className="text-blue-600 no-underline cursor-pointer font-semibold hover:text-blue-800"
        >
          [{n}]
        </a>
      );
    }
    return (
      <a
        href={href}
        className="text-blue-600 underline"
        target="_blank"
        rel="noreferrer"
      >
        {children}
      </a>
    );
  },
  strong: ({ children }: React.HTMLAttributes<HTMLElement>) => (
    <strong className="font-semibold">{children}</strong>
  ),
  blockquote: ({ children }: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote className="border-l-4 border-gray-300 pl-3 italic text-gray-600 mb-2">
      {children}
    </blockquote>
  ),
  // Fenced code blocks render as <pre><code>. The <pre> is the dark
  // container; the <code> inside should be transparent so we don't get
  // a nested background. We detect a block by the `language-*` class
  // that react-markdown adds to fenced code.
  pre: ({ children }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-3 font-mono text-sm">
      {children}
    </pre>
  ),
  code: ({ className, children }: React.HTMLAttributes<HTMLElement>) => {
    const isBlock = typeof className === "string" && /language-/.test(className);
    if (isBlock) {
      return (
        <code className="bg-transparent p-0 text-gray-100 font-mono">
          {children}
        </code>
      );
    }
    return (
      <code className="bg-gray-200 text-pink-600 px-1.5 py-0.5 rounded text-[0.85em] font-mono">
        {children}
      </code>
    );
  },
};

export default function Answer({ answer, isLoading, error }: AnswerProps) {
  if (error && !answer) {
    return (
      <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-yellow-800 text-sm">
          Answer could not be generated ({error}), but matching sources are
          shown below.
        </p>
      </div>
    );
  }

  if (!answer) {
    if (isLoading) {
      return (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-700">Generating answer…</p>
        </div>
      );
    }
    return null;
  }

  return (
    <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <h2 className="text-sm font-semibold text-blue-800 mb-2">
        Answer
        {isLoading && (
          <span className="ml-2 text-blue-400 animate-pulse">streaming…</span>
        )}
      </h2>
      <div className="text-gray-800 text-sm">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={mdComponents}
        >
          {linkifyCitations(answer)}
        </ReactMarkdown>
        {isLoading && (
          <span className="inline-block w-2 h-4 bg-blue-500 align-middle animate-pulse" />
        )}
      </div>
    </div>
  );
}
