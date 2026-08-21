FROM node:20-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Bake ML models into the image so cold starts don't re-download them.
# Set WARM_EMBEDDINGS=true / WARM_RERANKER=true to bake the local models
# (not needed when using EMBED_PROVIDER=gemini / RERANK_PROVIDER=jina).
ARG WARM_EMBEDDINGS=false
ARG WARM_RERANKER=false
ENV WARM_EMBEDDINGS=${WARM_EMBEDDINGS} WARM_RERANKER=${WARM_RERANKER}
# Always create .cache so the COPY below succeeds even when
# no models were baked in (API-only providers).
RUN mkdir -p .cache && node scripts/warm-models.mjs

FROM node:20-slim
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/.cache ./.cache
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "run", "start"]
