"""
Interactive HTML wrapper for TMAP SVG: magnifier, zoom control, nearest-molecule label,
and optional 2D structure (SMILES) via SmilesDrawer in the browser.

Layout: CSS **grid** — overview (small) and magnifier **panel** side by side on wide screens;
**stacked** on narrow viewports. Viewport meta + fluid gaps + ``min()`` / ``clamp()`` for
responsive sizing. Embedding-comparison blocks sit full-width below the grid.

``nodes`` entries should include ``x``, ``y``, ``label``; add ``smiles`` per node for
the structure panel (same order as TMAP vertices).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

# Logarithmic magnifier: linear slider 0..SLIDER_MAX → zoom ZOOM_MIN..ZOOM_MAX
_MAG_SLIDER_MAX = 1000
_MAG_ZOOM_MIN = 1.0
_MAG_ZOOM_MAX = 100.0

# Overview map = small context; magnifier panel = primary (large lens on screen).
_MAG_LEN_DISPLAY_PX = 420


def json_for_html_script(obj: Any) -> str:
    """JSON safe to embed in HTML (avoid ``</script>`` breaks)."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace("<", "\\u003c")


def merge_smiles_into_nodes(
    nodes: list[dict[str, Any]], smiles: list[str]
) -> list[dict[str, Any]]:
    """Return new node dicts with ``smiles`` set (aligned by index)."""
    if len(smiles) != len(nodes):
        raise ValueError(
            f"smiles length {len(smiles)} != nodes length {len(nodes)}"
        )
    return [{**n, "smiles": s} for n, s in zip(nodes, smiles, strict=True)]


def write_tmap_html_simple(
    out_html: Path,
    svg_content: str,
    *,
    page_title: str,
    extra_html: str | None = None,
) -> None:
    """Minimal HTML: embedded SVG only, optional ``extra_html`` before ``</body>``."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    title_esc = html.escape(page_title, quote=False)
    embed_style = ""
    if extra_html:
        embed_style = (
            ".tmap-embed-compare{margin:24px auto;max-width:min(1400px,98vw);padding:0 16px 32px;}"
            ".tmap-embed-compare h2{font-size:16px;font-weight:600;color:#333;margin:0 0 8px;}"
            ".tmap-embed-caption{font-size:12px;color:#666;margin:0 0 14px;line-height:1.45;}"
            ".tmap-embed-svg-wrap{width:100%;overflow:auto;background:#fff;border:1px solid #e0e0e0;"
            "border-radius:8px;padding:8px;}"
            ".tmap-embed-svg-wrap svg{max-width:100%;height:auto;display:block;}"
        )
    body = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        f"<title>{title_esc}</title>"
        "<style>body{margin:0;background:#fafafa;font-family:system-ui,sans-serif;}"
        "#tmap-shell{display:inline-block;max-width:100vw;margin:0 auto;}"
        "#tmap-root{display:block;max-width:100vw;max-height:100vh;}"
        f"{embed_style}</style></head><body>\n"
        '<div id="tmap-shell">\n'
        + svg_content
        + "\n</div>\n"
        + (extra_html or "")
        + "</body></html>\n"
    )
    out_html.write_text(body, encoding="utf-8")


# SmilesDrawer 1.x UMD: stable parse(smiles, success, err) → SVG string
SMILES_DRAWER_CDN = (
    "https://cdn.jsdelivr.net/npm/smiles-drawer@1.1.0/dist/smiles-drawer.min.js"
)


def write_tmap_html_interactive(
    out_html: Path,
    svg_content: str,
    *,
    page_title: str,
    nodes: list[dict[str, Any]],
    mag_zoom: float,
    smiles_drawer_url: str = SMILES_DRAWER_CDN,
    show_structure_viewer: bool = True,
    colorbar: dict[str, Any] | None = None,
    show_data_table: bool = True,
    magnifier_clone_graph: bool = True,
    extra_html_before_script: str | None = None,
) -> None:
    """
    Full viewer: magnifier with live **logarithmic** zoom slider (1×–100×),
    nearest ``label``, optional 2D structure when each node includes ``smiles``,
    optional **colorbar** in the side panel (uses SVG gradient ``#tmap-cbar-grad``),
    and an optional **filterable table** to jump the magnifier to a molecule.

    ``mag_zoom`` is the initial magnification in [1, 100], mapped to the slider
    position via ``100 ** (slider / 1000)``.

    ``colorbar`` should be ``{"vmin", "vmax", "label"}`` from
    ``build_annotated_tmap_svg`` (or ``None`` if no activity scale).

    Set ``magnifier_clone_graph=False`` to omit the magnifier's ``<use href="#tmap-graph"/>``
    (Chrome duplicates the whole graph in memory — disabling saves a lot on large maps).

    The data table is **lazy**: it does not scan or render rows until the user types a
    query of at least 2 characters, and it caps visible rows to limit DOM size.

    ``extra_html_before_script`` is inserted after the side panel and before the main
    script block (e.g. PCA / UMAP / t-SNE comparison figures).
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)
    title_esc = html.escape(page_title, quote=False)
    nodes_json = json_for_html_script(nodes)
    mag_lens_js = json.dumps(magnifier_clone_graph)
    mag_clamped = max(_MAG_ZOOM_MIN, min(_MAG_ZOOM_MAX, float(mag_zoom)))
    mag_ref_js = float(mag_clamped)
    initial_slider = int(
        round(
            _MAG_SLIDER_MAX
            * math.log(mag_clamped)
            / math.log(_MAG_ZOOM_MAX)
        )
    )
    initial_slider = max(0, min(_MAG_SLIDER_MAX, initial_slider))

    _lw = _MAG_LEN_DISPLAY_PX
    magnifier_svg_html = (
        f'<svg id="tmap-mag-svg" xmlns="http://www.w3.org/2000/svg" width="{_lw}" height="{_lw}" '
        'viewBox="0 0 8000 8000" preserveAspectRatio="xMidYMid meet">\n'
        '<use href="#tmap-graph"/>\n'
        "</svg>\n"
    )
    if not magnifier_clone_graph:
        magnifier_svg_html = (
            f'<svg id="tmap-mag-svg" xmlns="http://www.w3.org/2000/svg" width="{_lw}" height="{_lw}" '
            'viewBox="0 0 200 200" preserveAspectRatio="xMidYMid meet">\n'
            '<rect width="200" height="200" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>\n'
            '<text x="100" y="88" text-anchor="middle" font-size="12" fill="#555" '
            'font-family="system-ui,sans-serif">Lens off</text>\n'
            '<text x="100" y="108" text-anchor="middle" font-size="11" fill="#888" '
            'font-family="system-ui,sans-serif">(saves memory)</text>\n'
            "</svg>\n"
        )
    mag_hint_html = (
        '<p id="tmap-mag-hint">Magnifier (main view) — move over the <strong>small overview</strong> '
        "left. Nearest molecule:</p>\n"
        if magnifier_clone_graph
        else '<p id="tmap-mag-hint">Magnifier lens off (saves memory). Use the overview + table. Nearest:</p>\n'
    )

    colorbar_panel_html = ""
    if colorbar is not None:
        lab_s = html.escape(str(colorbar["label"]), quote=False)
        vmin_s = html.escape(f"{float(colorbar['vmin']):.4g}", quote=False)
        vmax_s = html.escape(f"{float(colorbar['vmax']):.4g}", quote=False)
        colorbar_panel_html = (
            '<div id="tmap-panel-cbar" class="tmap-panel-cbar" aria-label="Color scale">\n'
            f'<p class="tmap-section-title">{lab_s}</p>\n'
            '<div class="tmap-panel-cbar-row">\n'
            f'<span class="tmap-cbar-axis-max">{vmax_s}</span>\n'
            '<svg width="28" height="160" viewBox="0 0 28 160" '
            'class="tmap-cbar-strip" aria-hidden="true">\n'
            '<rect x="1" y="1" width="26" height="158" fill="url(#tmap-cbar-grad)" '
            'stroke="#222" stroke-width="1" rx="2"/>\n'
            "</svg>\n"
            f'<span class="tmap-cbar-axis-min">{vmin_s}</span>\n'
            "</div>\n"
            "</div>\n"
        )

    data_table_html = ""
    if show_data_table:
        data_table_html = (
            '<p class="tmap-section-title">Molecules</p>\n'
            '<input type="search" id="tmap-df-filter" class="tmap-df-filter" '
            'placeholder="Type ≥2 chars to search (name / value / SMILES)…" autocomplete="off"/>\n'
            '<p id="tmap-df-caption" class="tmap-df-caption"></p>\n'
            '<div class="tmap-df-scroll">\n'
            '<table class="tmap-df-table" id="tmap-df-table">\n'
            "<thead><tr><th>#</th><th>Label</th><th>Value</th><th>SMILES</th></tr></thead>\n"
            '<tbody id="tmap-df-tbody"></tbody>\n'
            "</table>\n"
            "</div>\n"
        )

    structure_block = ""
    draw_fn = ""
    if show_structure_viewer:
        structure_block = """
        <p class="tmap-section-title">Selected structure</p>
        <div id="tmap-mol-viewer" class="tmap-mol-viewer"></div>
        <pre id="tmap-mol-smiles" class="tmap-mol-smiles"></pre>
"""
        draw_fn = """
  const molHost = document.getElementById('tmap-mol-viewer');
  const smiEl = document.getElementById('tmap-mol-smiles');
  function drawSelectedMolecule(idx) {
    if (!molHost || !smiEl) return;
    const n = nodes[idx];
    const smi = n.smiles || '';
    smiEl.textContent = smi || '';
    molHost.innerHTML = '';
    if (!smi) {
      molHost.textContent = '—';
      return;
    }
    if (typeof SmilesDrawer !== 'undefined' && typeof SmilesDrawer.parse === 'function') {
      try {
        SmilesDrawer.parse(smi, function(svg) {
          molHost.innerHTML = svg;
        }, function() {
          molHost.innerHTML = '<span class="tmap-mol-fallback">(draw failed)</span>';
        });
      } catch (e) {
        molHost.innerHTML = '<pre class="tmap-mol-fallback">' + smi + '</pre>';
      }
    } else {
      molHost.innerHTML = '<pre class="tmap-mol-fallback">' + smi + '</pre>';
    }
  }
"""
        draw_call_update = "    drawSelectedMolecule(j);\n"
        draw_call_init = "    drawSelectedMolecule(j);\n"
        draw_call_focus = "    drawSelectedMolecule(j);\n"
    else:
        draw_call_update = ""
        draw_call_init = ""
        draw_call_focus = ""

    table_init_js = ""
    if show_data_table:
        table_init_js = r"""
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }
  function rowMatches(n, q) {
    const smi = (n.smiles || '').toLowerCase();
    const lab = String(n.label || '').toLowerCase();
    let valStr = '';
    const v = n.value;
    if (v !== undefined && v !== null && Number.isFinite(v)) valStr = String(v);
    return lab.includes(q) || smi.includes(q) || valStr.includes(q);
  }
  const TABLE_MIN_QUERY = 2;
  const TABLE_MAX_ROWS = 80;
  function renderMoleculeTable() {
    if (!tbody) return;
    const q = (filterEl && filterEl.value || '').trim().toLowerCase();
    if (q.length < TABLE_MIN_QUERY) {
      tbody.innerHTML = '';
      if (capEl) {
        capEl.textContent = 'Type at least ' + TABLE_MIN_QUERY + ' characters to search. ' +
          '(' + nodes.length + ' molecules — lazy load saves memory.)';
      }
      return;
    }
    const rows = [];
    for (let i = 0; i < nodes.length && rows.length < TABLE_MAX_ROWS; i++) {
      if (!rowMatches(nodes[i], q)) continue;
      rows.push(i);
    }
    if (capEl) {
      if (rows.length >= TABLE_MAX_ROWS) {
        capEl.textContent = 'First ' + TABLE_MAX_ROWS + ' matches — narrow your search.';
      } else {
        capEl.textContent = rows.length + ' match(es)';
      }
    }
    let html = '';
    for (let k = 0; k < rows.length; k++) {
      const i = rows[k];
      const n = nodes[i];
      const v = n.value;
      const vs = (v !== undefined && v !== null && Number.isFinite(v)) ? Number(v).toFixed(3) : '—';
      const smi = (n.smiles || '');
      const smiShort = smi.length > 56 ? smi.slice(0, 53) + '…' : smi;
      html += '<tr data-idx="' + i + '" tabindex="0" role="button"><td>' + i + '</td><td>' + escHtml(String(n.label || '')) + '</td><td class="tmap-df-num">' + vs + '</td><td class="tmap-df-smi" title="' + escHtml(smi) + '">' + escHtml(smiShort) + '</td></tr>';
    }
    tbody.innerHTML = html;
  }
"""

    ext_script = (
        f"""
(function() {{
  const nodes = JSON.parse(document.getElementById('tmap-node-data').textContent);
  const MAG_LENS = {mag_lens_js};
  const main = document.getElementById('tmap-root');
  const lens = document.getElementById('tmap-mag-svg');
  const labelEl = document.getElementById('tmap-mag-label');
  const zoomInput = document.getElementById('tmap-mag-zoom');
  const zoomVal = document.getElementById('tmap-mag-zoom-val');
  const SLIDER_MAX = {_MAG_SLIDER_MAX};
  const ZOOM_MAX = {_MAG_ZOOM_MAX};
  const magRef = {mag_ref_js};
  function sliderToZoom(s) {{
    const t = Math.max(0, Math.min(SLIDER_MAX, parseFloat(s))) / SLIDER_MAX;
    return Math.pow(ZOOM_MAX, t);
  }}
  let magZoom = sliderToZoom(zoomInput.value);
  let lastClient = null;
  const cell = Math.max(50, Math.min(400, (main.viewBox.baseVal.width + main.viewBox.baseVal.height) / 80));
  const grid = new Map();
  for (let i = 0; i < nodes.length; i++) {{
    const cx = Math.floor(nodes[i].x / cell);
    const cy = Math.floor(nodes[i].y / cell);
    const k = cx + ',' + cy;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }}
  function nearest(px, py) {{
    const cx = Math.floor(px / cell);
    const cy = Math.floor(py / cell);
    let best = 0, bestD = Infinity;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {{
      const arr = grid.get((cx+dx) + ',' + (cy+dy));
      if (!arr) continue;
      for (const i of arr) {{
        const d = (nodes[i].x - px) ** 2 + (nodes[i].y - py) ** 2;
        if (d < bestD) {{ bestD = d; best = i; }}
      }}
    }}
    if (bestD < Infinity) return best;
    for (let i = 0; i < nodes.length; i++) {{
      const d = (nodes[i].x - px) ** 2 + (nodes[i].y - py) ** 2;
      if (d < bestD) {{ bestD = d; best = i; }}
    }}
    return best;
  }}
  function svgPoint(svg, clientX, clientY) {{
    const pt = svg.createSVGPoint();
    pt.x = clientX; pt.y = clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }}
  function svgClientFromSvgCoords(svg, x, y) {{
    const pt = svg.createSVGPoint();
    pt.x = x; pt.y = y;
    return pt.matrixTransform(svg.getScreenCTM());
  }}
  const vb = main.viewBox.baseVal;
  const vbW = vb.width, vbH = vb.height;
  let raf = null;
  const tbody = document.getElementById('tmap-df-tbody');
  const capEl = document.getElementById('tmap-df-caption');
  const filterEl = document.getElementById('tmap-df-filter');
  function setNearestLabel(j) {{
    let t = nodes[j].label;
    const v = nodes[j].value;
    if (v !== undefined && v !== null && Number.isFinite(v)) t += ' — ' + Number(v).toFixed(2);
    labelEl.textContent = t;
  }}
{draw_fn}
  function focusNodeIndex(j) {{
    if (j < 0 || j >= nodes.length) return;
    const vw = vbW / magZoom, vh = vbH / magZoom;
    const nx = nodes[j].x, ny = nodes[j].y;
    if (MAG_LENS && lens) {{
      lens.setAttribute('viewBox', (nx - vw/2) + ' ' + (ny - vh/2) + ' ' + vw + ' ' + vh);
    }}
    setNearestLabel(j);
{draw_call_focus}    const lc = svgClientFromSvgCoords(main, nx, ny);
    lastClient = {{ x: lc.x, y: lc.y }};
    if (tbody) {{
      document.querySelectorAll('#tmap-df-tbody tr').forEach(function(tr) {{
        tr.classList.toggle('tmap-df-row-active', parseInt(tr.getAttribute('data-idx'), 10) === j);
      }});
    }}
  }}
"""
        + (table_init_js if show_data_table else "")
        + f"""
  function update(e) {{
    const src = e.touches ? e.touches[0] : e;
    lastClient = {{ x: src.clientX, y: src.clientY }};
    const p = svgPoint(main, src.clientX, src.clientY);
    if (p.x < vb.x || p.y < vb.y || p.x > vb.x + vbW || p.y > vb.y + vbH) return;
    const vw = vbW / magZoom, vh = vbH / magZoom;
    if (MAG_LENS && lens) {{
      lens.setAttribute('viewBox', (p.x - vw/2) + ' ' + (p.y - vh/2) + ' ' + vw + ' ' + vh);
    }}
    const j = nearest(p.x, p.y);
    setNearestLabel(j);
{draw_call_update}  }}
  function onPointer(e) {{
    if (raf) return;
    raf = requestAnimationFrame(() => {{ raf = null; update(e); }});
  }}
  main.addEventListener('mousemove', onPointer);
  main.addEventListener('touchmove', (e) => {{ e.preventDefault(); onPointer(e); }}, {{ passive: false }});
  function fmtZoom(z) {{
    if (z >= 10) return z.toFixed(z >= 100 ? 0 : 1);
    const r = Math.round(z * 100) / 100;
    return (r % 1 === 0) ? String(r) : r.toFixed(2);
  }}
  function applyZoom() {{
    magZoom = sliderToZoom(zoomInput.value);
    if (!Number.isFinite(magZoom) || magZoom < 1) magZoom = magRef;
    zoomVal.textContent = '×' + fmtZoom(magZoom);
    if (lastClient) {{
      update({{ clientX: lastClient.x, clientY: lastClient.y, touches: null }});
    }} else {{
      init();
    }}
  }}
  function init() {{
    magZoom = sliderToZoom(zoomInput.value);
    if (!Number.isFinite(magZoom) || magZoom < 1) magZoom = magRef;
    zoomVal.textContent = '×' + fmtZoom(magZoom);
    const br = main.getBoundingClientRect();
    const p = svgPoint(main, br.left + br.width / 2, br.top + br.height / 2);
    const vw = vbW / magZoom, vh = vbH / magZoom;
    if (MAG_LENS && lens) {{
      lens.setAttribute('viewBox', (p.x - vw/2) + ' ' + (p.y - vh/2) + ' ' + vw + ' ' + vh);
    }}
    const j = nearest(p.x, p.y);
    setNearestLabel(j);
{draw_call_init}  }}
  zoomInput.addEventListener('input', applyZoom);
  zoomInput.addEventListener('change', applyZoom);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
"""
        + (
            """
  let dfTimer = null;
  if (filterEl) {
    filterEl.addEventListener('input', function() {
      clearTimeout(dfTimer);
      dfTimer = setTimeout(renderMoleculeTable, 220);
    });
  }
  if (tbody) {
    tbody.addEventListener('click', function(e) {
      const tr = e.target.closest('tr[data-idx]');
      if (!tr) return;
      focusNodeIndex(parseInt(tr.getAttribute('data-idx'), 10));
    });
    tbody.addEventListener('keydown', function(e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const tr = e.target.closest('tr[data-idx]');
      if (!tr) return;
      e.preventDefault();
      focusNodeIndex(parseInt(tr.getAttribute('data-idx'), 10));
    });
    renderMoleculeTable();
  }
"""
            if show_data_table
            else ""
        )
        + """
})();
"""
    )

    sd_script = ""
    if show_structure_viewer:
        sd_script = (
            f'  <script src="{html.escape(smiles_drawer_url, quote=True)}"></script>\n'
        )

    body = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        f"<title>{title_esc}</title>"
        "<style>"
        ":root{"
        f"--tmap-lens-max:{_MAG_LEN_DISPLAY_PX}px;"
        "--tmap-overview-max-h:min(38vh,420px);"
        "--tmap-rail-pad:clamp(8px,2.5vw,20px);"
        "}"
        "*,*::before,*::after{box-sizing:border-box;}"
        "body{margin:0;background:#fafafa;font-family:system-ui,-apple-system,sans-serif;"
        "line-height:1.45;min-height:100dvh;}"
        ".tmap-layout{"
        "display:grid;grid-template-columns:minmax(0,min(42vw,440px)) minmax(260px,1fr);"
        "gap:clamp(10px,2vw,18px);align-items:start;"
        "padding:var(--tmap-rail-pad);padding-bottom:clamp(16px,4vh,32px);"
        "max-width:100%;width:100%;}"
        "#tmap-shell{position:relative;min-width:0;width:100%;"
        "border:1px solid #e0e0e0;border-radius:10px;background:#fff;"
        "box-shadow:0 2px 12px rgba(0,0,0,.07);padding:6px;}"
        "#tmap-root{display:block;max-width:100%;width:100%;height:auto;"
        "max-height:var(--tmap-overview-max-h);cursor:crosshair;margin:0 auto;}"
        "#tmap-mag-panel.tmap-panel{position:sticky;top:var(--tmap-rail-pad);"
        "z-index:10;width:100%;max-width:min(100%,560px);justify-self:end;"
        "max-height:min(92dvh,920px);overflow-x:hidden;overflow-y:auto;"
        "-webkit-overflow-scrolling:touch;"
        "background:#fff;border:1px solid #ccc;border-radius:10px;"
        "box-shadow:0 4px 20px rgba(0,0,0,.1);padding:clamp(10px,2.5vw,14px);}"
        "#tmap-mag-hint{font-size:11px;color:#666;margin:0 0 8px;line-height:1.35;}"
        ".tmap-section-title{font-size:11px;font-weight:600;color:#444;margin:10px 0 6px;}"
        "#tmap-mag-zoom-row{display:flex;align-items:center;gap:8px;margin:0 0 10px;flex-wrap:wrap;}"
        "#tmap-mag-zoom-row label{font-size:11px;color:#444;}"
        "#tmap-mag-zoom{flex:1;min-width:100px;max-width:160px;}"
        "#tmap-mag-zoom-val{font-size:12px;font-weight:600;color:#333;min-width:2.5em;}"
        "#tmap-mag-svg{width:min(100%,var(--tmap-lens-max));height:auto;"
        "aspect-ratio:1;display:block;border:1px solid #ddd;"
        "border-radius:6px;background:#fff;max-width:100%;}"
        "#tmap-mag-label{margin-top:8px;font-size:12px;color:#222;word-break:break-word;line-height:1.4;}"
        ".tmap-mol-viewer{min-height:180px;display:flex;align-items:center;justify-content:center;"
        "border:1px solid #e0e0e0;border-radius:4px;background:#fafafa;padding:6px;}"
        ".tmap-mol-viewer svg{max-width:100%;max-height:240px;height:auto;}"
        ".tmap-mol-smiles{font-size:10px;color:#555;white-space:pre-wrap;word-break:break-all;"
        "margin:6px 0 0;max-height:72px;overflow:auto;}"
        ".tmap-mol-fallback{font-size:10px;color:#333;}"
        ".tmap-panel-cbar{margin:0 0 10px;}"
        ".tmap-panel-cbar-row{display:flex;flex-direction:column;align-items:center;gap:4px;}"
        ".tmap-cbar-axis-max,.tmap-cbar-axis-min{font-size:10px;color:#444;"
        "font-variant-numeric:tabular-nums;line-height:1.2;}"
        ".tmap-cbar-strip{display:block;border-radius:2px;}"
        ".tmap-df-filter{width:100%;box-sizing:border-box;font-size:12px;padding:6px 8px;"
        "border:1px solid #ccc;border-radius:6px;margin:0 0 6px;}"
        ".tmap-df-caption{font-size:10px;color:#666;margin:0 0 6px;min-height:1.2em;}"
        ".tmap-df-scroll{max-height:min(320px,38vh);overflow:auto;border:1px solid #e0e0e0;"
        "border-radius:6px;background:#fafafa;}"
        ".tmap-df-table{width:100%;border-collapse:collapse;font-size:11px;}"
        ".tmap-df-table th{text-align:left;padding:6px 8px;background:#eee;position:sticky;"
        "top:0;z-index:1;font-weight:600;}"
        ".tmap-df-table td{padding:4px 8px;border-top:1px solid #e8e8e8;vertical-align:top;}"
        ".tmap-df-table tr{cursor:pointer;}"
        ".tmap-df-table tr:hover{background:#f0f7ff;}"
        ".tmap-df-row-active{background:#d6eaff !important;}"
        ".tmap-df-num{font-variant-numeric:tabular-nums;white-space:nowrap;color:#333;}"
        ".tmap-df-smi{font-size:10px;color:#555;word-break:break-all;}"
        ".tmap-below{width:100%;padding:0 var(--tmap-rail-pad) clamp(24px,5vh,48px);"
        "clear:both;}"
        ".tmap-embed-compare{margin:0 auto;max-width:min(1400px,100%);padding:0;}"
        ".tmap-embed-compare h2{font-size:16px;font-weight:600;color:#333;margin:0 0 8px;}"
        ".tmap-embed-caption{font-size:12px;color:#666;margin:0 0 14px;line-height:1.45;}"
        ".tmap-embed-svg-wrap{width:100%;overflow:auto;background:#fff;border:1px solid #e0e0e0;"
        "border-radius:8px;padding:8px;}"
        ".tmap-embed-svg-wrap svg{max-width:100%;height:auto;display:block;}"
        "@media (max-width:1100px){"
        ".tmap-layout{grid-template-columns:minmax(0,1fr) minmax(240px,1fr);"
        "gap:clamp(8px,1.5vw,14px);}"
        "}"
        "@media (max-width:820px){"
        ":root{--tmap-overview-max-h:min(42vh,380px);}"
        ".tmap-layout{grid-template-columns:1fr;}"
        "#tmap-mag-panel.tmap-panel{position:relative;top:0;max-height:none;"
        "max-width:100%;justify-self:stretch;}"
        "#tmap-shell{max-width:min(100%,520px);margin:0 auto;}"
        "}"
        "@media (max-width:480px){"
        ":root{--tmap-overview-max-h:min(36vh,320px);}"
        "#tmap-mag-zoom-row{flex-direction:column;align-items:stretch;}"
        "#tmap-mag-zoom{max-width:100%;}"
        ".tmap-df-scroll{max-height:min(220px,45dvh);}"
        "}"
        "</style>"
        f"{sd_script}"
        "</head><body>\n"
        f'<script type="application/json" id="tmap-node-data">{nodes_json}</script>\n'
        '<div class="tmap-layout">\n'
        '<div id="tmap-shell">\n'
        + svg_content
        + "\n</div>\n"
        '<aside id="tmap-mag-panel" class="tmap-panel">\n'
        + mag_hint_html
        + '<div id="tmap-mag-zoom-row">\n'
        '<label for="tmap-mag-zoom">Zoom (log 1–100×)</label>\n'
        f'<input type="range" id="tmap-mag-zoom" min="0" max="{_MAG_SLIDER_MAX}" step="1" value="{initial_slider}"/>\n'
        '<span id="tmap-mag-zoom-val"></span>\n'
        "</div>\n"
        + colorbar_panel_html
        + magnifier_svg_html
        + '<div id="tmap-mag-label">—</div>\n'
        + structure_block
        + data_table_html
        + "</aside>\n"
        "</div>\n"
        + (
            '<div class="tmap-below">\n' + extra_html_before_script + "</div>\n"
            if extra_html_before_script
            else ""
        )
        + f"<script>{ext_script}</script>\n"
        "</body></html>\n"
    )
    out_html.write_text(body, encoding="utf-8")
