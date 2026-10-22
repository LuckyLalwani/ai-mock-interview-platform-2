# ============================================================
# Stage 1: Build the React + TypeScript frontend
# ============================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first for better caching
COPY package.json package-lock.json ./
RUN npm ci

# Pass build-time environment variables for Vite
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ARG VITE_ML_SERVICE_URL

ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL \
    VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY \
    VITE_ML_SERVICE_URL=$VITE_ML_SERVICE_URL

# Copy source code and build
COPY . .
RUN npm run build

# ============================================================
# Stage 2: Serve with Nginx on Cloud Run
# ============================================================
FROM nginx:alpine

# Default PORT for Cloud Run
ENV PORT=8080

# Remove default nginx static assets and config
RUN rm -rf /usr/share/nginx/html/*

# Copy built SPA assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy Nginx template (nginx:alpine automatically substitutes ${PORT} using envsubst)
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
