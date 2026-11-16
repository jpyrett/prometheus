import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// Dev: Vite serves the page on 5173 and proxies /api to the Python server (default 127.0.0.1:8000).
// To point it somewhere else: PROMETHEUS_API=http://127.0.0.1:8100 npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const api = env.PROMETHEUS_API || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { "/api": { target: api, changeOrigin: false } } },
    preview: { port: 4173, proxy: { "/api": { target: api, changeOrigin: false } } },
    build: { outDir: "dist", sourcemap: false },
    test: { environment: "jsdom", globals: false, include: ["src/**/*.test.{ts,tsx}"], setupFiles: ["src/test/setup.ts"] },
  };
});
