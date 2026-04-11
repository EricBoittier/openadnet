import {
  copyFileSync,
  existsSync,
  mkdirSync,
  realpathSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** `file:`-linked @pymol-rs/viewer lives outside this folder; Vite must be allowed to read dist/*.wasm. */
function viewerPackageRealPaths() {
  const linked = resolve(__dirname, "node_modules/@pymol-rs/viewer");
  if (!existsSync(linked)) return [];
  try {
    return [realpathSync(linked)];
  } catch {
    return [linked];
  }
}

/**
 * WebAssembly.instantiateStreaming() requires Content-Type: application/wasm.
 * Some Vite code paths leave it empty; COEP also expects explicit CORP on assets.
 */
function wasmHeadersPlugin() {
  const middleware = (req, res, next) => {
    const pathOnly = req.url?.split("?")[0] ?? "";
    if (pathOnly.endsWith(".wasm")) {
      res.setHeader("Content-Type", "application/wasm");
      res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    }
    next();
  };

  function mountFirst(server) {
    try {
      const stack = server.middlewares?.stack;
      if (Array.isArray(stack)) {
        stack.unshift({ route: "", handle: middleware });
        return;
      }
    } catch {
      /* fall through */
    }
    server.middlewares.use(middleware);
  }

  return {
    name: "wasm-headers",
    enforce: "pre",
    configureServer(server) {
      mountFirst(server);
    },
    configurePreviewServer(server) {
      mountFirst(server);
    },
  };
}

/** The viewer chunk loads `pymol_web_bg.wasm` from the same directory as the chunk; ensure it exists after `vite build`. */
function copyViewerWasm() {
  return {
    name: "copy-viewer-wasm",
    closeBundle() {
      const src = resolve(
        "node_modules/@pymol-rs/viewer/dist/pymol_web_bg.wasm",
      );
      const destDir = resolve("dist/assets");
      if (!existsSync(src)) return;
      mkdirSync(destDir, { recursive: true });
      copyFileSync(src, resolve(destDir, "pymol_web_bg.wasm"));
    },
  };
}

export default defineConfig({
  plugins: [wasmHeadersPlugin(), copyViewerWasm()],
  // Pre-bundling can break wasm-bindgen's fetch + MIME expectations for this package.
  optimizeDeps: {
    exclude: ["@pymol-rs/viewer"],
  },
  server: {
    fs: {
      allow: [".", ...viewerPackageRealPaths()],
    },
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
