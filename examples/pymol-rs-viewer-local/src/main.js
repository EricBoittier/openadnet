import { PyMolRSViewer } from "@pymol-rs/viewer";

let viewerFailureUiShown = false;

const WGPU_CANVAS_CONTEXT_HINT =
  "This is a known Firefox + wgpu/WebGPU issue: the canvas context is valid in JavaScript but the WASM stack rejects it. Open this page in Chrome or Chromium instead (especially on Linux). See https://github.com/gfx-rs/wgpu/issues/8378";

function showFailureBox(title, extraLines = []) {
  if (viewerFailureUiShown) return;
  viewerFailureUiShown = true;
  const box = document.createElement("div");
  const t = document.createElement("p");
  t.style.cssText =
    "padding:1rem;color:#c00;max-width:42rem;line-height:1.4;margin:0;";
  t.textContent = title;
  box.appendChild(t);
  for (const line of extraLines) {
    const sub = document.createElement("p");
    sub.style.cssText =
      "padding:0 1rem 1rem;color:#333;max-width:42rem;line-height:1.4;font-size:0.9rem;margin:0;";
    sub.textContent = line;
    box.appendChild(sub);
  }
  document.body.prepend(box);
}

function guessFormat(fileName) {
  const lower = fileName.toLowerCase();
  const parts = lower.split(".").filter(Boolean);
  let ext = parts.at(-1);
  if (ext === "gz") ext = parts.at(-2);
  switch (ext) {
    case "cif":
    case "mmcif":
      return "cif";
    case "bcif":
      return "bcif";
    case "pdb":
    case "ent":
    default:
      return "pdb";
  }
}

function baseName(fileName) {
  let n = fileName;
  if (n.toLowerCase().endsWith(".gz")) n = n.slice(0, -3);
  n = n.replace(/\.[^.]+$/, "");
  return n || "structure";
}

async function main() {
  const root = document.getElementById("viewer");
  if (!root) throw new Error("#viewer missing");

  const viewer = new PyMolRSViewer(root);
  await viewer.init();

  document.getElementById("file")?.addEventListener("change", async (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement)) return;
    const file = input.files?.[0];
    if (!file) return;

    const data = new Uint8Array(await file.arrayBuffer());
    viewer.loadData(data, baseName(file.name), guessFormat(file.name));
    viewer.execute("show cartoon");
    viewer.execute("color spectrum");
    viewer.execute("zoom all");
  });
}

function handleViewerInitFailure(err) {
  console.error(err);
  const msg = String(err?.message ?? err);
  const lines = [];
  if (
    /GPUCanvasContext|canvas context is not a GPUCanvasContext/i.test(msg) ||
    (/unreachable executed/i.test(msg) && /Firefox/i.test(navigator.userAgent))
  ) {
    lines.push(WGPU_CANVAS_CONTEXT_HINT);
  } else if (/webgpu|WebGPU|GPUAdapter|wgpu|navigator\.gpu/i.test(msg)) {
    lines.push(
      "This app needs WebGPU. Try current Chrome or Edge. On Linux, prefer Chromium; Firefox WebGPU can still hit engine/wgpu bugs.",
    );
  }
  showFailureBox(`Failed to start viewer: ${msg}`, lines);
}

main().catch(handleViewerInitFailure);

// WASM panics may not reject the `init()` promise; surface the same hint in the console/UI.
window.addEventListener("error", (ev) => {
  const m = `${ev.message ?? ""} ${ev.error?.message ?? ""}`;
  if (/GPUCanvasContext|canvas context is not a GPUCanvasContext/i.test(m)) {
    showFailureBox(
      "Viewer crashed (WASM): canvas WebGPU context could not be used.",
      [WGPU_CANVAS_CONTEXT_HINT],
    );
  }
});
