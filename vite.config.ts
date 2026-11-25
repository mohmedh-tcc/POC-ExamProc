import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import fs from 'node:fs';

function servePublicWasmPlugin(): Plugin {
  return {
    name: 'serve-public-wasm',
    configureServer(server) {
      // Prepend middleware so it handles /wasm/ requests before Vite's transformMiddleware
      server.middlewares.use((req, res, next) => {
        const rawUrl = req.url || '';
        const [pathname] = rawUrl.split('?');
        if (pathname.startsWith('/wasm/')) {
          const filePath = path.join(__dirname, 'public', pathname);
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            if (pathname.endsWith('.js')) {
              res.setHeader('Content-Type', 'application/javascript');
            } else if (pathname.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
            } else if (pathname.endsWith('.onnx')) {
              res.setHeader('Content-Type', 'application/octet-stream');
            }
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            res.writeHead(200);
            fs.createReadStream(filePath).pipe(res);
            return;
          }
        }
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        const rawUrl = req.url || '';
        const [pathname] = rawUrl.split('?');
        if (pathname.startsWith('/wasm/')) {
          const filePath = path.join(__dirname, 'dist', pathname);
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            if (pathname.endsWith('.js')) {
              res.setHeader('Content-Type', 'application/javascript');
            } else if (pathname.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
            } else if (pathname.endsWith('.onnx')) {
              res.setHeader('Content-Type', 'application/octet-stream');
            }
            res.writeHead(200);
            fs.createReadStream(filePath).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), servePublicWasmPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  worker: {
    format: 'es',
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
