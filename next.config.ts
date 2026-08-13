import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  serverExternalPackages: [
    "@lancedb/lancedb",
    "@huggingface/transformers",
    "sharp",
    "onnxruntime-node",
  ],
};

export default nextConfig;
