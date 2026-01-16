# Multi-stage Dockerfile for MOTIS UI
# Stage 1: Build
FROM node:20-alpine AS builder

# Install pnpm directly (more reliable than corepack in Docker)
RUN npm install -g pnpm@latest

# Set working directory
WORKDIR /app

# Copy workspace configuration files
COPY pnpm-workspace.yaml pnpm-lock.yaml package.json ./

# Copy API package (workspace dependency)
COPY api/package.json api/tsconfig.json ./api/
COPY api/openapi/ ./api/openapi/

# Copy openapi.yaml (required for API client generation)
# The API build script references ../../openapi.yaml from api/ directory
# From /app/api/, ../../openapi.yaml points to /openapi.yaml (root)
# So we need to copy it to the root of the container, not /app/
COPY openapi.yaml /openapi.yaml

# Copy UI source files
COPY src/ ./src/
COPY static/ ./static/
COPY svelte.config.js vite.config.ts tsconfig.json tailwind.config.js postcss.config.js components.json ./

# Copy sprite directory (optional - will be skipped if not present via .dockerignore)
# Since the directory exists in this project, we copy it normally
COPY sprite/ ./sprite/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Build the API client first (workspace dependency)
RUN pnpm --filter @motis-project/motis-client build

# Build the UI
RUN pnpm build

# Stage 2: Production server
FROM nginx:alpine

# Install gettext for envsubst
RUN apk add --no-cache gettext

# Copy built files from builder
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration template
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set default backend URL as environment variable
ENV MOTIS_BACKEND_URL=http://localhost:80

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Use custom entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

