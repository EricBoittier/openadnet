# openadnet

PXR challenge baseline: molecular descriptors, sklearn regressors with cross-validation, optional plots (`viz`), and uncertainty helpers (`uncertainty`).

### Fingerprint grid

Descriptors are built for **fp sizes** `512, 1024, 2048, 4096`. **Morgan** fingerprints use radii **`0, 1, 2, 3`** (RDKit bond-radius / “hop” count), named `morgan_r{radius}_{bits|count}_{size}`. For each size, RDKit, atom pair, and topological torsion are also included — each as **binary** (`*_bits_{size}`) and **count** (`*_count_{size}`) via `GetFingerprintAsNumPy` / `GetCountFingerprintAsNumPy`, plus `rdkit_phys_props`. Per-molecule values are cached on disk under `~/.cache/openadnet/fingerprints` (override with `OPENADNET_FP_CACHE`; bump `FP_CACHE_VERSION` in `features_data.py` if definitions change).

### CV result cache

Completed `(descriptor, model)` cross-validation metrics are stored in `outputs/baseline_cv_cache.json` keyed by a hash of the training SMILES + target and the CV config. Reruns **skip** pairs already in the cache. Remove the file or change `CV_RESULT_CACHE_SCHEMA` in `baseline.py` to force a full recompute.

## Latest baseline CV

<!-- BASELINE_CV_TABLE_START -->

<div class="br-table-title br-align-center">Baseline cross-validation (pEC50)</div>
<table class="br-table br-style-academic">
<thead>
<tr>
<th scope="col" style="text-align:center;"><strong></strong></th>
<th scope="col" style="text-align:center;"><strong>descriptor</strong></th>
<th scope="col" style="text-align:center;"><strong>model</strong></th>
<th scope="col" style="text-align:center;"><strong>mean_rmse</strong></th>
<th scope="col" style="text-align:center;"><strong>std_rmse</strong></th>
<th scope="col" style="text-align:center;"><strong>mean_mae</strong></th>
<th scope="col" style="text-align:center;"><strong>std_mae</strong></th>
<th scope="col" style="text-align:center;"><strong>mean_r2</strong></th>
<th scope="col" style="text-align:center;"><strong>std_r2</strong></th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="text-align:left;"><strong>0</strong></th>
<td style="text-align:right;">morgan_bits_2048_r2</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7674</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.5794</td>
<td style="text-align:right;">0.0098</td>
<td style="text-align:right;">0.5303</td>
<td style="text-align:right;">0.0220</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>1</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7859</td>
<td style="text-align:right;">0.0146</td>
<td style="text-align:right;">0.5647</td>
<td style="text-align:right;">0.0089</td>
<td style="text-align:right;">0.5073</td>
<td style="text-align:right;">0.0267</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>2</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7913</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.5925</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.5007</td>
<td style="text-align:right;">0.0269</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>3</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7946</td>
<td style="text-align:right;">0.0094</td>
<td style="text-align:right;">0.5847</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.4961</td>
<td style="text-align:right;">0.0287</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>4</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7982</td>
<td style="text-align:right;">0.0174</td>
<td style="text-align:right;">0.5872</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.4924</td>
<td style="text-align:right;">0.0114</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>5</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7990</td>
<td style="text-align:right;">0.0198</td>
<td style="text-align:right;">0.5884</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.4903</td>
<td style="text-align:right;">0.0359</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>6</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8156</td>
<td style="text-align:right;">0.0117</td>
<td style="text-align:right;">0.6086</td>
<td style="text-align:right;">0.0084</td>
<td style="text-align:right;">0.4693</td>
<td style="text-align:right;">0.0299</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>7</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8217</td>
<td style="text-align:right;">0.0113</td>
<td style="text-align:right;">0.6219</td>
<td style="text-align:right;">0.0067</td>
<td style="text-align:right;">0.4616</td>
<td style="text-align:right;">0.0241</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>8</strong></th>
<td style="text-align:right;">morgan_bits_2048_r2</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8340</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.6308</td>
<td style="text-align:right;">0.0084</td>
<td style="text-align:right;">0.4452</td>
<td style="text-align:right;">0.0283</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>9</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8352</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.6228</td>
<td style="text-align:right;">0.0089</td>
<td style="text-align:right;">0.4440</td>
<td style="text-align:right;">0.0147</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>10</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8353</td>
<td style="text-align:right;">0.0086</td>
<td style="text-align:right;">0.6339</td>
<td style="text-align:right;">0.0080</td>
<td style="text-align:right;">0.4436</td>
<td style="text-align:right;">0.0238</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>11</strong></th>
<td style="text-align:right;">morgan_bits_2048_r2</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8380</td>
<td style="text-align:right;">0.0142</td>
<td style="text-align:right;">0.6180</td>
<td style="text-align:right;">0.0142</td>
<td style="text-align:right;">0.4402</td>
<td style="text-align:right;">0.0216</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>12</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8521</td>
<td style="text-align:right;">0.0098</td>
<td style="text-align:right;">0.6495</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.4211</td>
<td style="text-align:right;">0.0228</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>13</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8686</td>
<td style="text-align:right;">0.0203</td>
<td style="text-align:right;">0.6658</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3990</td>
<td style="text-align:right;">0.0147</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>14</strong></th>
<td style="text-align:right;">morgan_bits_2048_r2</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8781</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.6775</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.3858</td>
<td style="text-align:right;">0.0060</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>15</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8783</td>
<td style="text-align:right;">0.0112</td>
<td style="text-align:right;">0.6856</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.3848</td>
<td style="text-align:right;">0.0271</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>16</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8844</td>
<td style="text-align:right;">0.0144</td>
<td style="text-align:right;">0.6817</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.3765</td>
<td style="text-align:right;">0.0251</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>17</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8972</td>
<td style="text-align:right;">0.0175</td>
<td style="text-align:right;">0.6929</td>
<td style="text-align:right;">0.0164</td>
<td style="text-align:right;">0.3586</td>
<td style="text-align:right;">0.0179</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>18</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9064</td>
<td style="text-align:right;">0.0304</td>
<td style="text-align:right;">0.6803</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.3453</td>
<td style="text-align:right;">0.0311</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>19</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9322</td>
<td style="text-align:right;">0.0339</td>
<td style="text-align:right;">0.7147</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.3080</td>
<td style="text-align:right;">0.0254</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>20</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9475</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.7469</td>
<td style="text-align:right;">0.0199</td>
<td style="text-align:right;">0.2849</td>
<td style="text-align:right;">0.0118</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>21</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0301</td>
<td style="text-align:right;">0.0372</td>
<td style="text-align:right;">0.7712</td>
<td style="text-align:right;">0.0169</td>
<td style="text-align:right;">0.1539</td>
<td style="text-align:right;">0.0532</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>22</strong></th>
<td style="text-align:right;">morgan_bits_2048_r2</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.1858</td>
<td style="text-align:right;">0.0247</td>
<td style="text-align:right;">0.9211</td>
<td style="text-align:right;">0.0129</td>
<td style="text-align:right;">-0.1204</td>
<td style="text-align:right;">0.0301</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>23</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2427</td>
<td style="text-align:right;">0.0374</td>
<td style="text-align:right;">0.9731</td>
<td style="text-align:right;">0.0286</td>
<td style="text-align:right;">-0.2346</td>
<td style="text-align:right;">0.1128</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>24</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2625</td>
<td style="text-align:right;">0.0884</td>
<td style="text-align:right;">0.8920</td>
<td style="text-align:right;">0.0219</td>
<td style="text-align:right;">-0.2705</td>
<td style="text-align:right;">0.1310</td>
</tr>
</tbody>
</table>
<div class="br-table-caption br-align-left">5-fold CV on the training set; metrics are out-of-fold. Sorted by mean RMSE (lower is better).</div>

<!-- BASELINE_CV_TABLE_END -->

### Artifacts

After a run, see:

- `outputs/baseline_cv_results.csv` — full numeric results
- `outputs/baseline_cv_results.html` — the same table as standalone HTML (also embedded above via [better-tables](https://pypi.org/project/better-tables/))

### Run

```bash
cd openadnet
PYTHONPATH=src python src/score_data.py
```

Requires Hugging Face Hub access for the training CSV on first load (cached afterward). Set `HF_TOKEN` for higher rate limits.

### Custom estimators and CV

Use the same cross-validation loop as `score_data.py`, but pass your own **`regressors`** mapping (string label → unfitted scikit-learn regressor). Labels appear in the results table under `model`.

**Steps**

1. From the `openadnet` package directory, set `PYTHONPATH=src` so modules `baseline`, `load_data`, and `features_data` import correctly.
2. Load the training table and build RDKit molecules from `SMILES` (same pattern as `score_data.py`).
3. Build a **`regressors`** dict: keys are short names for reporting; values must be sklearn-style regressors implementing `fit` / `predict` (typically **unfitted** instances).
4. Optionally set **`descriptor_names`** to a subset of `list_descriptor_names()` from `features_data` (otherwise the full grid runs).
5. Optionally set **`BaselineCVConfig`** (`y_col`, `n_splits`, `shuffle`, `cv_random_state`, `model_random_state`).
6. Call **`run_baseline_cv(...)`** with your `regressors` and config. Use **`cv_cache_path`** pointing to a separate JSON file if you do not want custom runs mixed into the default `outputs/baseline_cv_cache.json`, or set **`use_cv_cache=False`** to always refit. **`show_progress=False`** disables tqdm.

**Pipeline wrapping:** `make_regressor_pipeline` in `baseline.py` prepends **`SimpleImputer`** and, only when the key is exactly **`ridge`**, **`elasticnet`**, or **`svr`**, a **`StandardScaler`**. Other keys get imputer + model only. If your estimator needs scaling under a different name, wrap it in a small sklearn **`Pipeline`** (e.g. `StandardScaler` then your regressor) and pass that as the dict value.

**Example**

```python
from pathlib import Path

from rdkit import Chem
from sklearn.linear_model import HuberRegressor

from baseline import BaselineCVConfig, run_baseline_cv
from load_data import train

mols = list(train["SMILES"].apply(Chem.MolFromSmiles))

regressors = {
    "huber": HuberRegressor(epsilon=1.35, max_iter=200),
}

results = run_baseline_cv(
    train,
    mols,
    descriptor_names=["morgan_r2_bits_2048", "rdkit_phys_props"],
    regressors=regressors,
    config=BaselineCVConfig(n_splits=5),
    cv_cache_path=Path("outputs/baseline_cv_custom.json"),
    use_cv_cache=True,
    show_progress=True,
)
print(results)
```

Run with: `PYTHONPATH=src python your_script.py` from the `openadnet` package root.
