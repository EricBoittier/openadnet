import { PyMolRSViewer } from "@pymol-rs/viewer";

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

main().catch((err) => {
  console.error(err);
  const msg = String(err?.message ?? err);
  const box = document.createElement("div");
  const t = document.createElement("p");
  t.style.cssText = "padding:1rem;color:#c00;max-width:42rem;line-height:1.4;margin:0;";
  t.textContent = `Failed to start viewer: ${msg}`;
  box.appendChild(t);
  if (/webgpu|WebGPU|GPUAdapter|wgpu|navigator\.gpu/i.test(msg)) {
    const sub = document.createElement("p");
    sub.style.cssText =
      "padding:0 1rem 1rem;color:#333;max-width:42rem;line-height:1.4;font-size:0.9rem;margin:0;";
    sub.textContent =
      "This app needs WebGPU. Try current Chrome or Edge, or Firefox with dom.webgpu.enabled (e.g. Nightly / beta). On Linux, use a GPU driver stack supported by your browser.";
    box.appendChild(sub);
  }
  document.body.prepend(box);
});
