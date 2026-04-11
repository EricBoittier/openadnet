# PyMOL-RS viewer — local file example

Minimal Vite app that embeds `@pymol-rs/viewer` and loads a structure from disk via **File** → **`loadData()`**.

## Prerequisites

- A checkout of **pymol-rs** with the **web** package built (`dist/` + `pkg/`).
- A **Chromium**-based browser (or other WebGPU-capable browser).

## One-time: build the viewer

From your **pymol-rs** tree (not this folder):

```bash
cd /path/to/pymol-rs/web
npm install
npm run build
```

(`build` runs `wasm-pack` and Vite library build.)

## Link to pymol-rs

`package.json` points at the viewer with:

```json
"@pymol-rs/viewer": "file:../../../../pymol-rs/web"
```

That assumes:

`~/Documents/openadnet/examples/pymol-rs-viewer-local` → `~/pymol-rs/web`

If your paths differ, edit that `file:` URL (or publish / link the package however you prefer).

## Run

```bash
cd examples/pymol-rs-viewer-local
npm install
npm run dev
```

Open the printed URL (e.g. `http://localhost:5173`), choose a `.pdb` / `.cif` / `.bcif` file.

Production static output:

```bash
npm run build
npm run preview
```

## Notes

- Vite sets **COOP** / **COEP** headers to match the viewer’s WASM expectations.
- The build copies `pymol_web_bg.wasm` next to the bundled JS so `import.meta.url` resolves correctly.
- If `npm install` fails on the `file:` path, fix the path or clone **pymol-rs** next to this repo and adjust `file:` accordingly.

### “unsupported MIME type '' expected application/wasm”

Use **`npm run dev`** or **`npm run preview`**, not `file://` or a static server that omits WASM MIME types (e.g. plain `python -m http.server`). This project’s Vite config forces **`Content-Type: application/wasm`**, skips pre-bundling the viewer package, and sets **`Cross-Origin-Resource-Policy`** for COEP. After changing `vite.config.js`, restart the dev server and hard-refresh the page. If it still fails, clear Vite’s cache: **`rm -rf node_modules/.vite`** then **`npm run dev`** again.
