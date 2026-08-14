import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  serverExternalPackages: [
    "@lancedb/lancedb",
    "@huggingface/transformers",
    "sharp",
    "onnxruntime-node",
    "pdf-parse",
    "@napi-rs/canvas",
  ],
};

export default nextConfig;
