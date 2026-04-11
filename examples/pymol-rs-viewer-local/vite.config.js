import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

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
  plugins: [copyViewerWasm()],
  server: {
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
