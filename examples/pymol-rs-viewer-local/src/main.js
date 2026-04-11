import { PyMolRSViewer } from "@pymol-rs/viewer";

function showWebGpuBanner(message) {
  const el = document.getElementById("webgpu-warning");
  if (!el) return;
  el.textContent = message;
  el.style.display = "block";
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
  if (!("gpu" in navigator)) {
    showWebGpuBanner(
      "WebGPU is not available in this browser. Try an up-to-date Chromium-based browser, or enable WebGPU in Firefox Nightly (about:config).",
    );
  }

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
  const t = document.createElement("p");
  t.textContent = `Failed to start viewer: ${err?.message ?? err}`;
  t.style.cssText = "padding:1rem;color:#c00;";
  document.body.prepend(t);
});
