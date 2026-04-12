# openadnet

Code and baselines for **PXR (pregnane X receptor) activity** modeling on the OpenADMET / Hugging Face challenge data: RDKit fingerprints plus classical regressors, optional **PyTorch Geometric** graph models, **HuggingFace** encoder regressors, **delta learning**, **ensembles**, **submission** CSV helpers, **uncertainty** (including conformal-style tooling), and **visualization** (e.g. TMAP, plots in `viz`).

| Topic | Where to start |
|--------|----------------|
| Install & optional DL stack | [Installation](#installation) |
| Sklearn CV grid & caching | [Fingerprint grid](#fingerprint-grid), [CV result cache](#cv-result-cache) |
| Full baseline leaderboard | [Latest baseline CV](#latest-baseline-cv) |
| PyG SMILES → graph → regression | [`docs/pyg_gnn.md`](docs/pyg_gnn.md) |
| Example PyG GNN CV | [PyG GNN (example run)](#pyg-gnn-example-run) |
| Challenge submission CSV | [`src/submission.py`](src/submission.py), [`scripts/write_submission.py`](scripts/write_submission.py) |

## Installation

From the repository root (Python **≥ 3.13**):

```bash
pip install -e .
```

Optional extras (see [`pyproject.toml`](pyproject.toml)):

```bash
pip install -e ".[dl]"     # torch, transformers, torch-geometric — GNNs & HF regressors
pip install -e ".[hydra]"  # Hydra for experiment sweeps (e.g. `scripts/hydra_gnn_sweep.py`)
```

Training data are loaded via Hugging Face Hub on first use and cached locally; set `HF_TOKEN` for higher rate limits.

## Repository layout (high level)

- **`src/baseline.py`**, **`src/features_data.py`**, **`src/score_data.py`** — descriptor grid, CV loop, default baseline run.
- **`src/models/`** — `GNNRegressor`, `HuggingFaceRegressor`, ensembles, `cv_dl` for neural CV; **`src/models/data/graph.py`** builds PyG graphs from SMILES.
- **`src/delta_learning.py`** — delta-learning helpers used with the baseline / DL stack.
- **`src/submission.py`** — validate predictions and write challenge-shaped submission files.
- **`src/uncertainty/`** — uncertainty and related plotting utilities.
- **`scripts/`** — CV drivers (`cv_gnn_regressor.py`, `cv_hf_regressor.py`, …), submission helpers, TMAP HTML, etc.
- **`examples/`** — small runnable snippets (HF CV, GNN subset, holdout fits).

### Fingerprint grid

Descriptors are built for **fp sizes** `512, 1024, 2048, 4096`. **Morgan** fingerprints use radii **`0, 1, 2, 3`** (RDKit bond-radius / “hop” count), named `morgan_r{radius}_{bits|count}_{size}`. For each size, RDKit, atom pair, and topological torsion are also included — each as **binary** (`*_bits_{size}`) and **count** (`*_count_{size}`) via `GetFingerprintAsNumPy` / `GetCountFingerprintAsNumPy`, plus `rdkit_phys_props`. Per-molecule values are cached on disk under `~/.cache/openadnet/fingerprints` (override with `OPENADNET_FP_CACHE`; bump `FP_CACHE_VERSION` in `features_data.py` if definitions change).

### CV result cache

Completed `(descriptor, model)` cross-validation metrics are stored in `outputs/baseline_cv_cache.json` keyed by a hash of the training SMILES + target and the CV config. Reruns **skip** pairs already in the cache. Remove the file or change `CV_RESULT_CACHE_SCHEMA` in `baseline.py` to force a full recompute.

### PyG GNN (example run)

Illustrative **GIN** encoder, **100** epochs per fold, **3-fold** CV (script default `--n-splits` unless overridden), other settings default (`scripts/cv_gnn_regressor.py`). Command: `PYTHONPATH=src python scripts/cv_gnn_regressor.py --epochs 100`.

| | mean RMSE | std RMSE | mean MAE | std MAE | mean R² | std R² |
|--|-----------|----------|----------|---------|---------|--------|
| **summary** | 0.8074 | 0.0241 | 0.6061 | 0.0103 | 0.4813 | 0.0156 |

Per-fold CSV: [`outputs/dl_cv_gnn_gin_e100_ns3.csv`](outputs/dl_cv_gnn_gin_e100_ns3.csv). This is **not** directly comparable row-for-row to the sklearn baseline table below (different fold count, model family, and training budget).

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
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7323</td>
<td style="text-align:right;">0.0244</td>
<td style="text-align:right;">0.5429</td>
<td style="text-align:right;">0.0227</td>
<td style="text-align:right;">0.5725</td>
<td style="text-align:right;">0.0198</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>1</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7334</td>
<td style="text-align:right;">0.0272</td>
<td style="text-align:right;">0.5423</td>
<td style="text-align:right;">0.0231</td>
<td style="text-align:right;">0.5712</td>
<td style="text-align:right;">0.0237</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>2</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7375</td>
<td style="text-align:right;">0.0270</td>
<td style="text-align:right;">0.5456</td>
<td style="text-align:right;">0.0222</td>
<td style="text-align:right;">0.5665</td>
<td style="text-align:right;">0.0210</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>3</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7389</td>
<td style="text-align:right;">0.0267</td>
<td style="text-align:right;">0.5464</td>
<td style="text-align:right;">0.0207</td>
<td style="text-align:right;">0.5648</td>
<td style="text-align:right;">0.0205</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>4</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7446</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.5560</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.5581</td>
<td style="text-align:right;">0.0137</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>5</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7446</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.5560</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.5581</td>
<td style="text-align:right;">0.0137</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>6</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7455</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.5590</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.5570</td>
<td style="text-align:right;">0.0111</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>7</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7456</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.5590</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.5569</td>
<td style="text-align:right;">0.0112</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>8</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7503</td>
<td style="text-align:right;">0.0315</td>
<td style="text-align:right;">0.5550</td>
<td style="text-align:right;">0.0259</td>
<td style="text-align:right;">0.5511</td>
<td style="text-align:right;">0.0278</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>9</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7504</td>
<td style="text-align:right;">0.0315</td>
<td style="text-align:right;">0.5550</td>
<td style="text-align:right;">0.0260</td>
<td style="text-align:right;">0.5511</td>
<td style="text-align:right;">0.0278</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>10</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7582</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.5654</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.5413</td>
<td style="text-align:right;">0.0257</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>11</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7583</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.5654</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.5412</td>
<td style="text-align:right;">0.0257</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>12</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7623</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.5683</td>
<td style="text-align:right;">0.0076</td>
<td style="text-align:right;">0.5363</td>
<td style="text-align:right;">0.0247</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>13</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7629</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.5684</td>
<td style="text-align:right;">0.0069</td>
<td style="text-align:right;">0.5356</td>
<td style="text-align:right;">0.0238</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>14</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7640</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.5687</td>
<td style="text-align:right;">0.0181</td>
<td style="text-align:right;">0.5347</td>
<td style="text-align:right;">0.0155</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>15</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7640</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.5687</td>
<td style="text-align:right;">0.0181</td>
<td style="text-align:right;">0.5346</td>
<td style="text-align:right;">0.0155</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>16</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7698</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.5828</td>
<td style="text-align:right;">0.0153</td>
<td style="text-align:right;">0.5273</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>17</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7698</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.5828</td>
<td style="text-align:right;">0.0153</td>
<td style="text-align:right;">0.5273</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>18</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7739</td>
<td style="text-align:right;">0.0226</td>
<td style="text-align:right;">0.5777</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5221</td>
<td style="text-align:right;">0.0282</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>19</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7739</td>
<td style="text-align:right;">0.0226</td>
<td style="text-align:right;">0.5777</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5221</td>
<td style="text-align:right;">0.0282</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>20</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7749</td>
<td style="text-align:right;">0.0239</td>
<td style="text-align:right;">0.5809</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5211</td>
<td style="text-align:right;">0.0266</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>21</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7749</td>
<td style="text-align:right;">0.0239</td>
<td style="text-align:right;">0.5809</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5211</td>
<td style="text-align:right;">0.0266</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>22</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7808</td>
<td style="text-align:right;">0.0257</td>
<td style="text-align:right;">0.5741</td>
<td style="text-align:right;">0.0179</td>
<td style="text-align:right;">0.5140</td>
<td style="text-align:right;">0.0221</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>23</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7826</td>
<td style="text-align:right;">0.0160</td>
<td style="text-align:right;">0.5901</td>
<td style="text-align:right;">0.0113</td>
<td style="text-align:right;">0.5114</td>
<td style="text-align:right;">0.0250</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>24</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7826</td>
<td style="text-align:right;">0.0160</td>
<td style="text-align:right;">0.5901</td>
<td style="text-align:right;">0.0113</td>
<td style="text-align:right;">0.5114</td>
<td style="text-align:right;">0.0250</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>25</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7845</td>
<td style="text-align:right;">0.0233</td>
<td style="text-align:right;">0.5885</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.5089</td>
<td style="text-align:right;">0.0294</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>26</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7846</td>
<td style="text-align:right;">0.0233</td>
<td style="text-align:right;">0.5884</td>
<td style="text-align:right;">0.0115</td>
<td style="text-align:right;">0.5087</td>
<td style="text-align:right;">0.0294</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>27</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7854</td>
<td style="text-align:right;">0.0314</td>
<td style="text-align:right;">0.5629</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.5084</td>
<td style="text-align:right;">0.0248</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>28</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7865</td>
<td style="text-align:right;">0.0208</td>
<td style="text-align:right;">0.5897</td>
<td style="text-align:right;">0.0155</td>
<td style="text-align:right;">0.5068</td>
<td style="text-align:right;">0.0187</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>29</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7891</td>
<td style="text-align:right;">0.0242</td>
<td style="text-align:right;">0.5813</td>
<td style="text-align:right;">0.0129</td>
<td style="text-align:right;">0.5037</td>
<td style="text-align:right;">0.0184</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>30</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7897</td>
<td style="text-align:right;">0.0232</td>
<td style="text-align:right;">0.5915</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.5024</td>
<td style="text-align:right;">0.0301</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>31</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7897</td>
<td style="text-align:right;">0.0232</td>
<td style="text-align:right;">0.5915</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.5024</td>
<td style="text-align:right;">0.0301</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>32</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7907</td>
<td style="text-align:right;">0.0302</td>
<td style="text-align:right;">0.5850</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.5014</td>
<td style="text-align:right;">0.0299</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>33</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7907</td>
<td style="text-align:right;">0.0296</td>
<td style="text-align:right;">0.5948</td>
<td style="text-align:right;">0.0235</td>
<td style="text-align:right;">0.5012</td>
<td style="text-align:right;">0.0319</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>34</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7907</td>
<td style="text-align:right;">0.0296</td>
<td style="text-align:right;">0.5948</td>
<td style="text-align:right;">0.0235</td>
<td style="text-align:right;">0.5012</td>
<td style="text-align:right;">0.0319</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>35</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7930</td>
<td style="text-align:right;">0.0168</td>
<td style="text-align:right;">0.5910</td>
<td style="text-align:right;">0.0164</td>
<td style="text-align:right;">0.4986</td>
<td style="text-align:right;">0.0153</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>36</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7951</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5967</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.4961</td>
<td style="text-align:right;">0.0138</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>37</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7955</td>
<td style="text-align:right;">0.0220</td>
<td style="text-align:right;">0.5987</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.4955</td>
<td style="text-align:right;">0.0178</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>38</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7961</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.5837</td>
<td style="text-align:right;">0.0120</td>
<td style="text-align:right;">0.4949</td>
<td style="text-align:right;">0.0120</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>39</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.7991</td>
<td style="text-align:right;">0.0256</td>
<td style="text-align:right;">0.6032</td>
<td style="text-align:right;">0.0157</td>
<td style="text-align:right;">0.4904</td>
<td style="text-align:right;">0.0333</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>40</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7996</td>
<td style="text-align:right;">0.0281</td>
<td style="text-align:right;">0.6048</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.4902</td>
<td style="text-align:right;">0.0283</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>41</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7999</td>
<td style="text-align:right;">0.0292</td>
<td style="text-align:right;">0.5920</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.4897</td>
<td style="text-align:right;">0.0292</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>42</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8008</td>
<td style="text-align:right;">0.0284</td>
<td style="text-align:right;">0.6060</td>
<td style="text-align:right;">0.0156</td>
<td style="text-align:right;">0.4886</td>
<td style="text-align:right;">0.0278</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>43</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8012</td>
<td style="text-align:right;">0.0258</td>
<td style="text-align:right;">0.6064</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.4876</td>
<td style="text-align:right;">0.0344</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>44</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8013</td>
<td style="text-align:right;">0.0175</td>
<td style="text-align:right;">0.5852</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.4881</td>
<td style="text-align:right;">0.0152</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>45</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8018</td>
<td style="text-align:right;">0.0263</td>
<td style="text-align:right;">0.5936</td>
<td style="text-align:right;">0.0215</td>
<td style="text-align:right;">0.4874</td>
<td style="text-align:right;">0.0258</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>46</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8027</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.5906</td>
<td style="text-align:right;">0.0085</td>
<td style="text-align:right;">0.4863</td>
<td style="text-align:right;">0.0148</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>47</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8028</td>
<td style="text-align:right;">0.0272</td>
<td style="text-align:right;">0.5905</td>
<td style="text-align:right;">0.0183</td>
<td style="text-align:right;">0.4862</td>
<td style="text-align:right;">0.0241</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>48</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8036</td>
<td style="text-align:right;">0.0317</td>
<td style="text-align:right;">0.5937</td>
<td style="text-align:right;">0.0217</td>
<td style="text-align:right;">0.4851</td>
<td style="text-align:right;">0.0301</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>49</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8038</td>
<td style="text-align:right;">0.0237</td>
<td style="text-align:right;">0.6067</td>
<td style="text-align:right;">0.0179</td>
<td style="text-align:right;">0.4851</td>
<td style="text-align:right;">0.0174</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>50</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8049</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.6070</td>
<td style="text-align:right;">0.0112</td>
<td style="text-align:right;">0.4834</td>
<td style="text-align:right;">0.0195</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>51</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8049</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.6070</td>
<td style="text-align:right;">0.0112</td>
<td style="text-align:right;">0.4834</td>
<td style="text-align:right;">0.0195</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>52</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8087</td>
<td style="text-align:right;">0.0251</td>
<td style="text-align:right;">0.5966</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.4786</td>
<td style="text-align:right;">0.0232</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>53</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8115</td>
<td style="text-align:right;">0.0278</td>
<td style="text-align:right;">0.6122</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.4746</td>
<td style="text-align:right;">0.0330</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>54</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8121</td>
<td style="text-align:right;">0.0235</td>
<td style="text-align:right;">0.5979</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.4741</td>
<td style="text-align:right;">0.0238</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>55</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8122</td>
<td style="text-align:right;">0.0262</td>
<td style="text-align:right;">0.6133</td>
<td style="text-align:right;">0.0194</td>
<td style="text-align:right;">0.4739</td>
<td style="text-align:right;">0.0274</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>56</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8134</td>
<td style="text-align:right;">0.0247</td>
<td style="text-align:right;">0.5969</td>
<td style="text-align:right;">0.0176</td>
<td style="text-align:right;">0.4726</td>
<td style="text-align:right;">0.0202</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>57</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8153</td>
<td style="text-align:right;">0.0264</td>
<td style="text-align:right;">0.6158</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.4700</td>
<td style="text-align:right;">0.0245</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>58</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8158</td>
<td style="text-align:right;">0.0274</td>
<td style="text-align:right;">0.6008</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.4696</td>
<td style="text-align:right;">0.0205</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>59</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8158</td>
<td style="text-align:right;">0.0275</td>
<td style="text-align:right;">0.6008</td>
<td style="text-align:right;">0.0159</td>
<td style="text-align:right;">0.4696</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>60</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8159</td>
<td style="text-align:right;">0.0273</td>
<td style="text-align:right;">0.6008</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.4696</td>
<td style="text-align:right;">0.0205</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>61</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8159</td>
<td style="text-align:right;">0.0274</td>
<td style="text-align:right;">0.6008</td>
<td style="text-align:right;">0.0159</td>
<td style="text-align:right;">0.4695</td>
<td style="text-align:right;">0.0205</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>62</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8167</td>
<td style="text-align:right;">0.0254</td>
<td style="text-align:right;">0.5899</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.4684</td>
<td style="text-align:right;">0.0190</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>63</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8167</td>
<td style="text-align:right;">0.0254</td>
<td style="text-align:right;">0.5899</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.4684</td>
<td style="text-align:right;">0.0190</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>64</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8167</td>
<td style="text-align:right;">0.0288</td>
<td style="text-align:right;">0.6168</td>
<td style="text-align:right;">0.0156</td>
<td style="text-align:right;">0.4681</td>
<td style="text-align:right;">0.0276</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>65</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8175</td>
<td style="text-align:right;">0.0362</td>
<td style="text-align:right;">0.6085</td>
<td style="text-align:right;">0.0216</td>
<td style="text-align:right;">0.4671</td>
<td style="text-align:right;">0.0361</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>66</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8190</td>
<td style="text-align:right;">0.0255</td>
<td style="text-align:right;">0.6200</td>
<td style="text-align:right;">0.0168</td>
<td style="text-align:right;">0.4651</td>
<td style="text-align:right;">0.0264</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>67</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8213</td>
<td style="text-align:right;">0.0246</td>
<td style="text-align:right;">0.6189</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.4624</td>
<td style="text-align:right;">0.0190</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>68</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8217</td>
<td style="text-align:right;">0.0291</td>
<td style="text-align:right;">0.5942</td>
<td style="text-align:right;">0.0183</td>
<td style="text-align:right;">0.4619</td>
<td style="text-align:right;">0.0248</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>69</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8241</td>
<td style="text-align:right;">0.0245</td>
<td style="text-align:right;">0.6220</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.4586</td>
<td style="text-align:right;">0.0232</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>70</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8244</td>
<td style="text-align:right;">0.0258</td>
<td style="text-align:right;">0.6111</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.4580</td>
<td style="text-align:right;">0.0269</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>71</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8248</td>
<td style="text-align:right;">0.0215</td>
<td style="text-align:right;">0.6081</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.4576</td>
<td style="text-align:right;">0.0218</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>72</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8249</td>
<td style="text-align:right;">0.0265</td>
<td style="text-align:right;">0.6218</td>
<td style="text-align:right;">0.0224</td>
<td style="text-align:right;">0.4575</td>
<td style="text-align:right;">0.0260</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>73</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8250</td>
<td style="text-align:right;">0.0176</td>
<td style="text-align:right;">0.6009</td>
<td style="text-align:right;">0.0148</td>
<td style="text-align:right;">0.4572</td>
<td style="text-align:right;">0.0191</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>74</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8262</td>
<td style="text-align:right;">0.0239</td>
<td style="text-align:right;">0.6100</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.4560</td>
<td style="text-align:right;">0.0199</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>75</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8262</td>
<td style="text-align:right;">0.0239</td>
<td style="text-align:right;">0.6100</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.4559</td>
<td style="text-align:right;">0.0199</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>76</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8262</td>
<td style="text-align:right;">0.0289</td>
<td style="text-align:right;">0.6228</td>
<td style="text-align:right;">0.0209</td>
<td style="text-align:right;">0.4557</td>
<td style="text-align:right;">0.0284</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>77</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8262</td>
<td style="text-align:right;">0.0289</td>
<td style="text-align:right;">0.6228</td>
<td style="text-align:right;">0.0209</td>
<td style="text-align:right;">0.4557</td>
<td style="text-align:right;">0.0284</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>78</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8265</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.6187</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.4551</td>
<td style="text-align:right;">0.0245</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>79</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8280</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.6215</td>
<td style="text-align:right;">0.0172</td>
<td style="text-align:right;">0.4533</td>
<td style="text-align:right;">0.0218</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>80</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8283</td>
<td style="text-align:right;">0.0246</td>
<td style="text-align:right;">0.6306</td>
<td style="text-align:right;">0.0160</td>
<td style="text-align:right;">0.4529</td>
<td style="text-align:right;">0.0270</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>81</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8287</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.6007</td>
<td style="text-align:right;">0.0150</td>
<td style="text-align:right;">0.4524</td>
<td style="text-align:right;">0.0191</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>82</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8290</td>
<td style="text-align:right;">0.0242</td>
<td style="text-align:right;">0.6187</td>
<td style="text-align:right;">0.0115</td>
<td style="text-align:right;">0.4523</td>
<td style="text-align:right;">0.0177</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>83</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8294</td>
<td style="text-align:right;">0.0320</td>
<td style="text-align:right;">0.6146</td>
<td style="text-align:right;">0.0221</td>
<td style="text-align:right;">0.4515</td>
<td style="text-align:right;">0.0316</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>84</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8299</td>
<td style="text-align:right;">0.0241</td>
<td style="text-align:right;">0.6191</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.4511</td>
<td style="text-align:right;">0.0174</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>85</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8305</td>
<td style="text-align:right;">0.0318</td>
<td style="text-align:right;">0.6295</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.4502</td>
<td style="text-align:right;">0.0295</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>86</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8305</td>
<td style="text-align:right;">0.0318</td>
<td style="text-align:right;">0.6295</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.4502</td>
<td style="text-align:right;">0.0295</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>87</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8344</td>
<td style="text-align:right;">0.0234</td>
<td style="text-align:right;">0.6266</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.4448</td>
<td style="text-align:right;">0.0239</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>88</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8345</td>
<td style="text-align:right;">0.0208</td>
<td style="text-align:right;">0.6272</td>
<td style="text-align:right;">0.0121</td>
<td style="text-align:right;">0.4444</td>
<td style="text-align:right;">0.0296</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>89</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8345</td>
<td style="text-align:right;">0.0208</td>
<td style="text-align:right;">0.6272</td>
<td style="text-align:right;">0.0121</td>
<td style="text-align:right;">0.4444</td>
<td style="text-align:right;">0.0296</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>90</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8348</td>
<td style="text-align:right;">0.0252</td>
<td style="text-align:right;">0.6255</td>
<td style="text-align:right;">0.0096</td>
<td style="text-align:right;">0.4445</td>
<td style="text-align:right;">0.0207</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>91</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8356</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.6305</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.4433</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>92</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8356</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.6305</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.4433</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>93</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8360</td>
<td style="text-align:right;">0.0282</td>
<td style="text-align:right;">0.6296</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.4426</td>
<td style="text-align:right;">0.0294</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>94</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8361</td>
<td style="text-align:right;">0.0260</td>
<td style="text-align:right;">0.6169</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.4426</td>
<td style="text-align:right;">0.0265</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>95</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8378</td>
<td style="text-align:right;">0.0104</td>
<td style="text-align:right;">0.6311</td>
<td style="text-align:right;">0.0071</td>
<td style="text-align:right;">0.4404</td>
<td style="text-align:right;">0.0130</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>96</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8389</td>
<td style="text-align:right;">0.0285</td>
<td style="text-align:right;">0.6160</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.4389</td>
<td style="text-align:right;">0.0276</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>97</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8397</td>
<td style="text-align:right;">0.0219</td>
<td style="text-align:right;">0.6366</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.4377</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>98</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8397</td>
<td style="text-align:right;">0.0219</td>
<td style="text-align:right;">0.6366</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.4377</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>99</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8398</td>
<td style="text-align:right;">0.0263</td>
<td style="text-align:right;">0.6226</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.4376</td>
<td style="text-align:right;">0.0264</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>100</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8404</td>
<td style="text-align:right;">0.0287</td>
<td style="text-align:right;">0.6218</td>
<td style="text-align:right;">0.0174</td>
<td style="text-align:right;">0.4369</td>
<td style="text-align:right;">0.0292</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>101</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8414</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.6122</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.4356</td>
<td style="text-align:right;">0.0197</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>102</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8422</td>
<td style="text-align:right;">0.0232</td>
<td style="text-align:right;">0.6369</td>
<td style="text-align:right;">0.0168</td>
<td style="text-align:right;">0.4346</td>
<td style="text-align:right;">0.0217</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>103</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8432</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.6209</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.4329</td>
<td style="text-align:right;">0.0234</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>104</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8438</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.6180</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.4322</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>105</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8440</td>
<td style="text-align:right;">0.0249</td>
<td style="text-align:right;">0.6208</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.4322</td>
<td style="text-align:right;">0.0224</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>106</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8456</td>
<td style="text-align:right;">0.0260</td>
<td style="text-align:right;">0.6297</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.4301</td>
<td style="text-align:right;">0.0210</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>107</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8459</td>
<td style="text-align:right;">0.0301</td>
<td style="text-align:right;">0.6501</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.4297</td>
<td style="text-align:right;">0.0267</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>108</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8500</td>
<td style="text-align:right;">0.0318</td>
<td style="text-align:right;">0.6278</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.4237</td>
<td style="text-align:right;">0.0341</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>109</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8502</td>
<td style="text-align:right;">0.0258</td>
<td style="text-align:right;">0.6368</td>
<td style="text-align:right;">0.0194</td>
<td style="text-align:right;">0.4239</td>
<td style="text-align:right;">0.0221</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>110</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8510</td>
<td style="text-align:right;">0.0361</td>
<td style="text-align:right;">0.6352</td>
<td style="text-align:right;">0.0235</td>
<td style="text-align:right;">0.4226</td>
<td style="text-align:right;">0.0361</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>111</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8512</td>
<td style="text-align:right;">0.0210</td>
<td style="text-align:right;">0.6482</td>
<td style="text-align:right;">0.0122</td>
<td style="text-align:right;">0.4223</td>
<td style="text-align:right;">0.0222</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>112</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8525</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.6419</td>
<td style="text-align:right;">0.0222</td>
<td style="text-align:right;">0.4198</td>
<td style="text-align:right;">0.0387</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>113</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8527</td>
<td style="text-align:right;">0.0111</td>
<td style="text-align:right;">0.6478</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.4204</td>
<td style="text-align:right;">0.0122</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>114</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8535</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.6494</td>
<td style="text-align:right;">0.0111</td>
<td style="text-align:right;">0.4191</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>115</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8544</td>
<td style="text-align:right;">0.0184</td>
<td style="text-align:right;">0.6493</td>
<td style="text-align:right;">0.0126</td>
<td style="text-align:right;">0.4176</td>
<td style="text-align:right;">0.0271</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>116</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8553</td>
<td style="text-align:right;">0.0294</td>
<td style="text-align:right;">0.6573</td>
<td style="text-align:right;">0.0177</td>
<td style="text-align:right;">0.4169</td>
<td style="text-align:right;">0.0257</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>117</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8562</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.6483</td>
<td style="text-align:right;">0.0093</td>
<td style="text-align:right;">0.4147</td>
<td style="text-align:right;">0.0346</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>118</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8566</td>
<td style="text-align:right;">0.0123</td>
<td style="text-align:right;">0.6527</td>
<td style="text-align:right;">0.0099</td>
<td style="text-align:right;">0.4151</td>
<td style="text-align:right;">0.0101</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>119</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8595</td>
<td style="text-align:right;">0.0317</td>
<td style="text-align:right;">0.6520</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.4108</td>
<td style="text-align:right;">0.0353</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>120</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8595</td>
<td style="text-align:right;">0.0317</td>
<td style="text-align:right;">0.6520</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.4108</td>
<td style="text-align:right;">0.0353</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>121</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8608</td>
<td style="text-align:right;">0.0155</td>
<td style="text-align:right;">0.6597</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.4092</td>
<td style="text-align:right;">0.0146</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>122</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8642</td>
<td style="text-align:right;">0.0276</td>
<td style="text-align:right;">0.6347</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.4045</td>
<td style="text-align:right;">0.0292</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>123</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8649</td>
<td style="text-align:right;">0.0311</td>
<td style="text-align:right;">0.6603</td>
<td style="text-align:right;">0.0182</td>
<td style="text-align:right;">0.4038</td>
<td style="text-align:right;">0.0278</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>124</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8654</td>
<td style="text-align:right;">0.0113</td>
<td style="text-align:right;">0.6633</td>
<td style="text-align:right;">0.0099</td>
<td style="text-align:right;">0.4031</td>
<td style="text-align:right;">0.0067</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>125</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8657</td>
<td style="text-align:right;">0.0172</td>
<td style="text-align:right;">0.6631</td>
<td style="text-align:right;">0.0095</td>
<td style="text-align:right;">0.4027</td>
<td style="text-align:right;">0.0083</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>126</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8659</td>
<td style="text-align:right;">0.0085</td>
<td style="text-align:right;">0.6622</td>
<td style="text-align:right;">0.0057</td>
<td style="text-align:right;">0.4020</td>
<td style="text-align:right;">0.0186</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>127</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8689</td>
<td style="text-align:right;">0.0244</td>
<td style="text-align:right;">0.6646</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3980</td>
<td style="text-align:right;">0.0276</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>128</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8703</td>
<td style="text-align:right;">0.0134</td>
<td style="text-align:right;">0.6679</td>
<td style="text-align:right;">0.0140</td>
<td style="text-align:right;">0.3963</td>
<td style="text-align:right;">0.0059</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>129</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8718</td>
<td style="text-align:right;">0.0372</td>
<td style="text-align:right;">0.6447</td>
<td style="text-align:right;">0.0236</td>
<td style="text-align:right;">0.3941</td>
<td style="text-align:right;">0.0379</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>130</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8750</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.6761</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.3896</td>
<td style="text-align:right;">0.0113</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>131</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8753</td>
<td style="text-align:right;">0.0142</td>
<td style="text-align:right;">0.6714</td>
<td style="text-align:right;">0.0091</td>
<td style="text-align:right;">0.3892</td>
<td style="text-align:right;">0.0142</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>132</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8766</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.6734</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.3865</td>
<td style="text-align:right;">0.0385</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>133</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8768</td>
<td style="text-align:right;">0.0251</td>
<td style="text-align:right;">0.6826</td>
<td style="text-align:right;">0.0131</td>
<td style="text-align:right;">0.3873</td>
<td style="text-align:right;">0.0200</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>134</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8773</td>
<td style="text-align:right;">0.0104</td>
<td style="text-align:right;">0.6770</td>
<td style="text-align:right;">0.0094</td>
<td style="text-align:right;">0.3865</td>
<td style="text-align:right;">0.0079</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>135</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8773</td>
<td style="text-align:right;">0.0150</td>
<td style="text-align:right;">0.6769</td>
<td style="text-align:right;">0.0141</td>
<td style="text-align:right;">0.3866</td>
<td style="text-align:right;">0.0081</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>136</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8782</td>
<td style="text-align:right;">0.0139</td>
<td style="text-align:right;">0.6775</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3853</td>
<td style="text-align:right;">0.0052</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>137</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8826</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.6732</td>
<td style="text-align:right;">0.0073</td>
<td style="text-align:right;">0.3786</td>
<td style="text-align:right;">0.0278</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>138</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8828</td>
<td style="text-align:right;">0.0243</td>
<td style="text-align:right;">0.6817</td>
<td style="text-align:right;">0.0177</td>
<td style="text-align:right;">0.3785</td>
<td style="text-align:right;">0.0288</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>139</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.8835</td>
<td style="text-align:right;">0.0283</td>
<td style="text-align:right;">0.6689</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3775</td>
<td style="text-align:right;">0.0315</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>140</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8835</td>
<td style="text-align:right;">0.0283</td>
<td style="text-align:right;">0.6689</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3775</td>
<td style="text-align:right;">0.0315</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>141</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8847</td>
<td style="text-align:right;">0.0232</td>
<td style="text-align:right;">0.6822</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3758</td>
<td style="text-align:right;">0.0282</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>142</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8852</td>
<td style="text-align:right;">0.0208</td>
<td style="text-align:right;">0.6793</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.3751</td>
<td style="text-align:right;">0.0243</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>143</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8893</td>
<td style="text-align:right;">0.0255</td>
<td style="text-align:right;">0.6627</td>
<td style="text-align:right;">0.0154</td>
<td style="text-align:right;">0.3697</td>
<td style="text-align:right;">0.0231</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>144</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8900</td>
<td style="text-align:right;">0.0173</td>
<td style="text-align:right;">0.6834</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.3684</td>
<td style="text-align:right;">0.0201</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>145</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8907</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.6897</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.3676</td>
<td style="text-align:right;">0.0123</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>146</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8921</td>
<td style="text-align:right;">0.0255</td>
<td style="text-align:right;">0.6885</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.3656</td>
<td style="text-align:right;">0.0211</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>147</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8934</td>
<td style="text-align:right;">0.0207</td>
<td style="text-align:right;">0.7007</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.3636</td>
<td style="text-align:right;">0.0244</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>148</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8936</td>
<td style="text-align:right;">0.0109</td>
<td style="text-align:right;">0.6896</td>
<td style="text-align:right;">0.0096</td>
<td style="text-align:right;">0.3634</td>
<td style="text-align:right;">0.0093</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>149</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8951</td>
<td style="text-align:right;">0.0255</td>
<td style="text-align:right;">0.6946</td>
<td style="text-align:right;">0.0173</td>
<td style="text-align:right;">0.3611</td>
<td style="text-align:right;">0.0308</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>150</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8958</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.6686</td>
<td style="text-align:right;">0.0082</td>
<td style="text-align:right;">0.3602</td>
<td style="text-align:right;">0.0185</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>151</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8958</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.6686</td>
<td style="text-align:right;">0.0082</td>
<td style="text-align:right;">0.3602</td>
<td style="text-align:right;">0.0185</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>152</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8977</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.6992</td>
<td style="text-align:right;">0.0111</td>
<td style="text-align:right;">0.3575</td>
<td style="text-align:right;">0.0194</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>153</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8987</td>
<td style="text-align:right;">0.0325</td>
<td style="text-align:right;">0.6937</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.3561</td>
<td style="text-align:right;">0.0340</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>154</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9007</td>
<td style="text-align:right;">0.0175</td>
<td style="text-align:right;">0.6854</td>
<td style="text-align:right;">0.0119</td>
<td style="text-align:right;">0.3523</td>
<td style="text-align:right;">0.0391</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>155</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9018</td>
<td style="text-align:right;">0.0452</td>
<td style="text-align:right;">0.6695</td>
<td style="text-align:right;">0.0302</td>
<td style="text-align:right;">0.3516</td>
<td style="text-align:right;">0.0473</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>156</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9024</td>
<td style="text-align:right;">0.0282</td>
<td style="text-align:right;">0.6767</td>
<td style="text-align:right;">0.0125</td>
<td style="text-align:right;">0.3511</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>157</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9058</td>
<td style="text-align:right;">0.0344</td>
<td style="text-align:right;">0.6832</td>
<td style="text-align:right;">0.0263</td>
<td style="text-align:right;">0.3462</td>
<td style="text-align:right;">0.0282</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>158</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9060</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.7057</td>
<td style="text-align:right;">0.0157</td>
<td style="text-align:right;">0.3459</td>
<td style="text-align:right;">0.0049</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>159</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9065</td>
<td style="text-align:right;">0.0212</td>
<td style="text-align:right;">0.7065</td>
<td style="text-align:right;">0.0140</td>
<td style="text-align:right;">0.3450</td>
<td style="text-align:right;">0.0160</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>160</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9092</td>
<td style="text-align:right;">0.0297</td>
<td style="text-align:right;">0.6914</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3409</td>
<td style="text-align:right;">0.0317</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>161</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.9092</td>
<td style="text-align:right;">0.0297</td>
<td style="text-align:right;">0.6914</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.3409</td>
<td style="text-align:right;">0.0317</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>162</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9109</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.7010</td>
<td style="text-align:right;">0.0122</td>
<td style="text-align:right;">0.3386</td>
<td style="text-align:right;">0.0165</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>163</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9117</td>
<td style="text-align:right;">0.0148</td>
<td style="text-align:right;">0.6840</td>
<td style="text-align:right;">0.0064</td>
<td style="text-align:right;">0.3374</td>
<td style="text-align:right;">0.0139</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>164</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9120</td>
<td style="text-align:right;">0.0311</td>
<td style="text-align:right;">0.6979</td>
<td style="text-align:right;">0.0190</td>
<td style="text-align:right;">0.3371</td>
<td style="text-align:right;">0.0283</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>165</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9129</td>
<td style="text-align:right;">0.0087</td>
<td style="text-align:right;">0.7015</td>
<td style="text-align:right;">0.0052</td>
<td style="text-align:right;">0.3349</td>
<td style="text-align:right;">0.0309</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>166</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9140</td>
<td style="text-align:right;">0.0178</td>
<td style="text-align:right;">0.7049</td>
<td style="text-align:right;">0.0115</td>
<td style="text-align:right;">0.3339</td>
<td style="text-align:right;">0.0205</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>167</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9159</td>
<td style="text-align:right;">0.0085</td>
<td style="text-align:right;">0.7069</td>
<td style="text-align:right;">0.0145</td>
<td style="text-align:right;">0.3308</td>
<td style="text-align:right;">0.0255</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>168</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9198</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.7174</td>
<td style="text-align:right;">0.0173</td>
<td style="text-align:right;">0.3257</td>
<td style="text-align:right;">0.0093</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>169</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9226</td>
<td style="text-align:right;">0.0322</td>
<td style="text-align:right;">0.7086</td>
<td style="text-align:right;">0.0193</td>
<td style="text-align:right;">0.3216</td>
<td style="text-align:right;">0.0295</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>170</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9257</td>
<td style="text-align:right;">0.0272</td>
<td style="text-align:right;">0.7093</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.3170</td>
<td style="text-align:right;">0.0254</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>171</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.9270</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.7033</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3148</td>
<td style="text-align:right;">0.0227</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>172</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.9270</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.7033</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3148</td>
<td style="text-align:right;">0.0227</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>173</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9270</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.7033</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3148</td>
<td style="text-align:right;">0.0227</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>174</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9270</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.7033</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.3148</td>
<td style="text-align:right;">0.0227</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>175</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9271</td>
<td style="text-align:right;">0.0228</td>
<td style="text-align:right;">0.7054</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.3144</td>
<td style="text-align:right;">0.0340</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>176</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9294</td>
<td style="text-align:right;">0.0259</td>
<td style="text-align:right;">0.6842</td>
<td style="text-align:right;">0.0229</td>
<td style="text-align:right;">0.3114</td>
<td style="text-align:right;">0.0253</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>177</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9294</td>
<td style="text-align:right;">0.0259</td>
<td style="text-align:right;">0.6842</td>
<td style="text-align:right;">0.0229</td>
<td style="text-align:right;">0.3114</td>
<td style="text-align:right;">0.0253</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>178</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9296</td>
<td style="text-align:right;">0.0527</td>
<td style="text-align:right;">0.6604</td>
<td style="text-align:right;">0.0265</td>
<td style="text-align:right;">0.3104</td>
<td style="text-align:right;">0.0626</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>179</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9350</td>
<td style="text-align:right;">0.0201</td>
<td style="text-align:right;">0.7282</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.3031</td>
<td style="text-align:right;">0.0196</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>180</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9350</td>
<td style="text-align:right;">0.0201</td>
<td style="text-align:right;">0.7282</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.3031</td>
<td style="text-align:right;">0.0196</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>181</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9380</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.6949</td>
<td style="text-align:right;">0.0102</td>
<td style="text-align:right;">0.2981</td>
<td style="text-align:right;">0.0295</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>182</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9381</td>
<td style="text-align:right;">0.0209</td>
<td style="text-align:right;">0.7335</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.2985</td>
<td style="text-align:right;">0.0177</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>183</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9396</td>
<td style="text-align:right;">0.0298</td>
<td style="text-align:right;">0.6917</td>
<td style="text-align:right;">0.0247</td>
<td style="text-align:right;">0.2962</td>
<td style="text-align:right;">0.0291</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>184</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">lgbm</td>
<td style="text-align:right;">0.9397</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.7130</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.2959</td>
<td style="text-align:right;">0.0259</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>185</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9397</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.7130</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.2959</td>
<td style="text-align:right;">0.0259</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>186</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9431</td>
<td style="text-align:right;">0.0319</td>
<td style="text-align:right;">0.7249</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.2913</td>
<td style="text-align:right;">0.0275</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>187</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9432</td>
<td style="text-align:right;">0.0325</td>
<td style="text-align:right;">0.6815</td>
<td style="text-align:right;">0.0227</td>
<td style="text-align:right;">0.2906</td>
<td style="text-align:right;">0.0381</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>188</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9449</td>
<td style="text-align:right;">0.0213</td>
<td style="text-align:right;">0.7444</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.2884</td>
<td style="text-align:right;">0.0186</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>189</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9459</td>
<td style="text-align:right;">0.0172</td>
<td style="text-align:right;">0.7385</td>
<td style="text-align:right;">0.0098</td>
<td style="text-align:right;">0.2868</td>
<td style="text-align:right;">0.0179</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>190</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9463</td>
<td style="text-align:right;">0.0238</td>
<td style="text-align:right;">0.7471</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.2863</td>
<td style="text-align:right;">0.0182</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>191</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9469</td>
<td style="text-align:right;">0.0156</td>
<td style="text-align:right;">0.7381</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.2852</td>
<td style="text-align:right;">0.0181</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>192</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9469</td>
<td style="text-align:right;">0.0156</td>
<td style="text-align:right;">0.7381</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.2852</td>
<td style="text-align:right;">0.0181</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>193</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9471</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.7460</td>
<td style="text-align:right;">0.0177</td>
<td style="text-align:right;">0.2850</td>
<td style="text-align:right;">0.0182</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>194</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9476</td>
<td style="text-align:right;">0.0154</td>
<td style="text-align:right;">0.7465</td>
<td style="text-align:right;">0.0125</td>
<td style="text-align:right;">0.2841</td>
<td style="text-align:right;">0.0152</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>195</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9486</td>
<td style="text-align:right;">0.0203</td>
<td style="text-align:right;">0.7469</td>
<td style="text-align:right;">0.0144</td>
<td style="text-align:right;">0.2828</td>
<td style="text-align:right;">0.0132</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>196</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9569</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.7554</td>
<td style="text-align:right;">0.0136</td>
<td style="text-align:right;">0.2700</td>
<td style="text-align:right;">0.0160</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>197</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9576</td>
<td style="text-align:right;">0.0137</td>
<td style="text-align:right;">0.7175</td>
<td style="text-align:right;">0.0084</td>
<td style="text-align:right;">0.2682</td>
<td style="text-align:right;">0.0355</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>198</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9600</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.7396</td>
<td style="text-align:right;">0.0138</td>
<td style="text-align:right;">0.2652</td>
<td style="text-align:right;">0.0230</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>199</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9608</td>
<td style="text-align:right;">0.0255</td>
<td style="text-align:right;">0.7321</td>
<td style="text-align:right;">0.0199</td>
<td style="text-align:right;">0.2639</td>
<td style="text-align:right;">0.0323</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>200</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9616</td>
<td style="text-align:right;">0.0164</td>
<td style="text-align:right;">0.7527</td>
<td style="text-align:right;">0.0177</td>
<td style="text-align:right;">0.2629</td>
<td style="text-align:right;">0.0186</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>201</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9619</td>
<td style="text-align:right;">0.0301</td>
<td style="text-align:right;">0.7397</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.2627</td>
<td style="text-align:right;">0.0255</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>202</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9672</td>
<td style="text-align:right;">0.0290</td>
<td style="text-align:right;">0.7393</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.2529</td>
<td style="text-align:right;">0.0543</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>203</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9687</td>
<td style="text-align:right;">0.0176</td>
<td style="text-align:right;">0.7662</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.2520</td>
<td style="text-align:right;">0.0125</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>204</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9731</td>
<td style="text-align:right;">0.0218</td>
<td style="text-align:right;">0.7668</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.2454</td>
<td style="text-align:right;">0.0143</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>205</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9731</td>
<td style="text-align:right;">0.0218</td>
<td style="text-align:right;">0.7668</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.2454</td>
<td style="text-align:right;">0.0143</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>206</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9821</td>
<td style="text-align:right;">0.0210</td>
<td style="text-align:right;">0.7469</td>
<td style="text-align:right;">0.0139</td>
<td style="text-align:right;">0.2312</td>
<td style="text-align:right;">0.0192</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>207</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9827</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.7472</td>
<td style="text-align:right;">0.0134</td>
<td style="text-align:right;">0.2303</td>
<td style="text-align:right;">0.0177</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>208</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9858</td>
<td style="text-align:right;">0.0194</td>
<td style="text-align:right;">0.7797</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.2256</td>
<td style="text-align:right;">0.0104</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>209</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9864</td>
<td style="text-align:right;">0.0709</td>
<td style="text-align:right;">0.6934</td>
<td style="text-align:right;">0.0150</td>
<td style="text-align:right;">0.2160</td>
<td style="text-align:right;">0.1398</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>210</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9928</td>
<td style="text-align:right;">0.0274</td>
<td style="text-align:right;">0.7671</td>
<td style="text-align:right;">0.0244</td>
<td style="text-align:right;">0.2139</td>
<td style="text-align:right;">0.0381</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>211</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9941</td>
<td style="text-align:right;">0.0214</td>
<td style="text-align:right;">0.7583</td>
<td style="text-align:right;">0.0155</td>
<td style="text-align:right;">0.2123</td>
<td style="text-align:right;">0.0186</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>212</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0357</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.7682</td>
<td style="text-align:right;">0.0106</td>
<td style="text-align:right;">0.1444</td>
<td style="text-align:right;">0.0319</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>213</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0421</td>
<td style="text-align:right;">0.0567</td>
<td style="text-align:right;">0.7626</td>
<td style="text-align:right;">0.0092</td>
<td style="text-align:right;">0.1320</td>
<td style="text-align:right;">0.0933</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>214</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0662</td>
<td style="text-align:right;">0.1056</td>
<td style="text-align:right;">0.7262</td>
<td style="text-align:right;">0.0252</td>
<td style="text-align:right;">0.0849</td>
<td style="text-align:right;">0.1897</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>215</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.1638</td>
<td style="text-align:right;">0.0253</td>
<td style="text-align:right;">0.9014</td>
<td style="text-align:right;">0.0243</td>
<td style="text-align:right;">-0.0810</td>
<td style="text-align:right;">0.0624</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>216</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.1690</td>
<td style="text-align:right;">0.0380</td>
<td style="text-align:right;">0.8936</td>
<td style="text-align:right;">0.0230</td>
<td style="text-align:right;">-0.0914</td>
<td style="text-align:right;">0.0837</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>217</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2494</td>
<td style="text-align:right;">0.0484</td>
<td style="text-align:right;">0.8942</td>
<td style="text-align:right;">0.0319</td>
<td style="text-align:right;">-0.2441</td>
<td style="text-align:right;">0.0630</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>218</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2535</td>
<td style="text-align:right;">0.0145</td>
<td style="text-align:right;">0.9890</td>
<td style="text-align:right;">0.0121</td>
<td style="text-align:right;">-0.2542</td>
<td style="text-align:right;">0.0653</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>219</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2821</td>
<td style="text-align:right;">0.0794</td>
<td style="text-align:right;">0.9611</td>
<td style="text-align:right;">0.0505</td>
<td style="text-align:right;">-0.3157</td>
<td style="text-align:right;">0.1655</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>220</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.4100</td>
<td style="text-align:right;">0.1809</td>
<td style="text-align:right;">0.8723</td>
<td style="text-align:right;">0.0313</td>
<td style="text-align:right;">-0.6126</td>
<td style="text-align:right;">0.3883</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>221</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">2.2302</td>
<td style="text-align:right;">1.1855</td>
<td style="text-align:right;">1.0332</td>
<td style="text-align:right;">0.0668</td>
<td style="text-align:right;">-4.1773</td>
<td style="text-align:right;">6.0527</td>
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

To refresh the leaderboard table **from cached JSON only** (no new CV runs), use
`scripts/readme_from_baseline_cache.py` — it fails if any pair is missing from
`outputs/baseline_cv_cache.json` (use `--fill-missing` to compute missing pairs first).

Requires Hugging Face Hub access for the training CSV on first load (cached afterward). Set `HF_TOKEN` for higher rate limits.

Graph- and transformer-based CV (optional `openadnet[dl]`) is wired through `scripts/cv_gnn_regressor.py`, `scripts/cv_hf_regressor.py`, and related drivers under `scripts/`; see [`docs/pyg_gnn.md`](docs/pyg_gnn.md) for the PyG pipeline. PyG GNN CV defaults to **50** epochs per fold (`--epochs` to change); use `--cpu` if CUDA runs out of memory.

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
