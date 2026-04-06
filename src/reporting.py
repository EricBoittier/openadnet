from __future__ import annotations

from pathlib import Path

import pandas as pd
from better_tables import build_table

BASELINE_TABLE_START = "<!-- BASELINE_CV_TABLE_START -->"
BASELINE_TABLE_END = "<!-- BASELINE_CV_TABLE_END -->"


def _format_results_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ("descriptor", "model"):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: round(float(x), 6) if pd.notna(x) else x)
    return out


def write_baseline_cv_artifacts(
    results: pd.DataFrame,
    project_root: Path,
    *,
    csv_name: str = "baseline_cv_results.csv",
    html_name: str = "baseline_cv_results.html",
    update_readme: bool = True,
) -> tuple[Path, Path | None]:
    """
    Write CSV under ``project_root/outputs/``, render HTML with better-tables,
    and inject HTML into ``README.md`` between marker comments (if present).
    """
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    results.to_csv(csv_path, index=False)

    display_df = _format_results_for_display(results)
    table = build_table(
        df=display_df,
        title="Baseline cross-validation (pEC50)",
        caption=(
            "5-fold CV on the training set; metrics are out-of-fold. "
            "Sorted by mean RMSE (lower is better)."
        ),
        style="academic",
        default_format={"kind": "number", "decimals": 4},
    )
    html = table.to_html()

    readme_path: Path | None = None
    if update_readme:
        readme_path = project_root / "README.md"
        if readme_path.is_file():
            _inject_html_into_readme(readme_path, html)
        else:
            readme_path.write_text(
                _default_readme_body(html),
                encoding="utf-8",
            )

    html_path = out_dir / html_name
    html_path.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>Baseline CV</title></head><body>\n"
        + html
        + "\n</body></html>\n",
        encoding="utf-8",
    )

    return csv_path, readme_path


def _inject_html_into_readme(readme_path: Path, html_fragment: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if BASELINE_TABLE_START not in text or BASELINE_TABLE_END not in text:
        raise ValueError(
            f"{readme_path} must contain {BASELINE_TABLE_START!r} and {BASELINE_TABLE_END!r}"
        )
    i0 = text.index(BASELINE_TABLE_START) + len(BASELINE_TABLE_START)
    i1 = text.index(BASELINE_TABLE_END)
    new_block = "\n\n" + html_fragment.strip() + "\n\n"
    readme_path.write_text(text[:i0] + new_block + text[i1:], encoding="utf-8")


def _default_readme_body(html: str) -> str:
    return f"""# openadnet

PXR challenge baseline utilities.

## Latest baseline CV

{BASELINE_TABLE_START}

{html.strip()}

{BASELINE_TABLE_END}

Results are also saved under `outputs/baseline_cv_results.csv` and `outputs/baseline_cv_results.html`.

Regenerate with:

```bash
cd openadnet
PYTHONPATH=src python src/score_data.py
```
"""
