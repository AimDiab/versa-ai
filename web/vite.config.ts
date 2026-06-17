import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiUrl = env.VITE_API_URL ?? "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 3000,
      // Dev-only proxy: routes /api/* to the FastAPI server so the browser
      // never has to deal with CORS during local development.
      proxy: {
        "/api": {
          target: apiUrl,
          changeOrigin: true,
        },
      },
    },
  };
});
