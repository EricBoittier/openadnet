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
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7341</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.5464</td>
<td style="text-align:right;">0.0148</td>
<td style="text-align:right;">0.5700</td>
<td style="text-align:right;">0.0273</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>1</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7354</td>
<td style="text-align:right;">0.0186</td>
<td style="text-align:right;">0.5451</td>
<td style="text-align:right;">0.0159</td>
<td style="text-align:right;">0.5685</td>
<td style="text-align:right;">0.0264</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>2</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7416</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.5545</td>
<td style="text-align:right;">0.0134</td>
<td style="text-align:right;">0.5615</td>
<td style="text-align:right;">0.0202</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>3</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7434</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.5578</td>
<td style="text-align:right;">0.0131</td>
<td style="text-align:right;">0.5593</td>
<td style="text-align:right;">0.0228</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>4</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7497</td>
<td style="text-align:right;">0.0184</td>
<td style="text-align:right;">0.5556</td>
<td style="text-align:right;">0.0180</td>
<td style="text-align:right;">0.5515</td>
<td style="text-align:right;">0.0272</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>5</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7582</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.5666</td>
<td style="text-align:right;">0.0176</td>
<td style="text-align:right;">0.5417</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>6</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7601</td>
<td style="text-align:right;">0.0104</td>
<td style="text-align:right;">0.5678</td>
<td style="text-align:right;">0.0124</td>
<td style="text-align:right;">0.5394</td>
<td style="text-align:right;">0.0172</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>7</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7645</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.5739</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.5340</td>
<td style="text-align:right;">0.0220</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>8</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7674</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.5794</td>
<td style="text-align:right;">0.0098</td>
<td style="text-align:right;">0.5303</td>
<td style="text-align:right;">0.0220</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>9</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7694</td>
<td style="text-align:right;">0.0051</td>
<td style="text-align:right;">0.5773</td>
<td style="text-align:right;">0.0064</td>
<td style="text-align:right;">0.5278</td>
<td style="text-align:right;">0.0231</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>10</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7755</td>
<td style="text-align:right;">0.0149</td>
<td style="text-align:right;">0.5695</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.5208</td>
<td style="text-align:right;">0.0159</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>11</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7766</td>
<td style="text-align:right;">0.0079</td>
<td style="text-align:right;">0.5864</td>
<td style="text-align:right;">0.0079</td>
<td style="text-align:right;">0.5192</td>
<td style="text-align:right;">0.0169</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>12</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7802</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.5915</td>
<td style="text-align:right;">0.0107</td>
<td style="text-align:right;">0.5146</td>
<td style="text-align:right;">0.0223</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>13</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7823</td>
<td style="text-align:right;">0.0221</td>
<td style="text-align:right;">0.5848</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.5123</td>
<td style="text-align:right;">0.0215</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>14</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7831</td>
<td style="text-align:right;">0.0180</td>
<td style="text-align:right;">0.5763</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.5115</td>
<td style="text-align:right;">0.0110</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>15</strong></th>
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
<th scope="row" style="text-align:left;"><strong>16</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7868</td>
<td style="text-align:right;">0.0019</td>
<td style="text-align:right;">0.5830</td>
<td style="text-align:right;">0.0026</td>
<td style="text-align:right;">0.5063</td>
<td style="text-align:right;">0.0215</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>17</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7868</td>
<td style="text-align:right;">0.0209</td>
<td style="text-align:right;">0.5921</td>
<td style="text-align:right;">0.0116</td>
<td style="text-align:right;">0.5062</td>
<td style="text-align:right;">0.0294</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>18</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7896</td>
<td style="text-align:right;">0.0223</td>
<td style="text-align:right;">0.5897</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.5030</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>19</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7907</td>
<td style="text-align:right;">0.0189</td>
<td style="text-align:right;">0.5932</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.5014</td>
<td style="text-align:right;">0.0266</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>20</strong></th>
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
<th scope="row" style="text-align:left;"><strong>21</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7913</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.5926</td>
<td style="text-align:right;">0.0160</td>
<td style="text-align:right;">0.5007</td>
<td style="text-align:right;">0.0230</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>22</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7946</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.5847</td>
<td style="text-align:right;">0.0084</td>
<td style="text-align:right;">0.4962</td>
<td style="text-align:right;">0.0287</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>23</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.7955</td>
<td style="text-align:right;">0.0016</td>
<td style="text-align:right;">0.5890</td>
<td style="text-align:right;">0.0031</td>
<td style="text-align:right;">0.4952</td>
<td style="text-align:right;">0.0235</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>24</strong></th>
<td style="text-align:right;">rdkit_phys_props</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.7960</td>
<td style="text-align:right;">0.0218</td>
<td style="text-align:right;">0.5843</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.4936</td>
<td style="text-align:right;">0.0420</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>25</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.7974</td>
<td style="text-align:right;">0.0190</td>
<td style="text-align:right;">0.5997</td>
<td style="text-align:right;">0.0115</td>
<td style="text-align:right;">0.4931</td>
<td style="text-align:right;">0.0238</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>26</strong></th>
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
<th scope="row" style="text-align:left;"><strong>27</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8010</td>
<td style="text-align:right;">0.0061</td>
<td style="text-align:right;">0.5925</td>
<td style="text-align:right;">0.0071</td>
<td style="text-align:right;">0.4884</td>
<td style="text-align:right;">0.0203</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>28</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8015</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.5953</td>
<td style="text-align:right;">0.0131</td>
<td style="text-align:right;">0.4882</td>
<td style="text-align:right;">0.0087</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>29</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8027</td>
<td style="text-align:right;">0.0232</td>
<td style="text-align:right;">0.5907</td>
<td style="text-align:right;">0.0217</td>
<td style="text-align:right;">0.4867</td>
<td style="text-align:right;">0.0189</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>30</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8027</td>
<td style="text-align:right;">0.0133</td>
<td style="text-align:right;">0.6054</td>
<td style="text-align:right;">0.0110</td>
<td style="text-align:right;">0.4860</td>
<td style="text-align:right;">0.0275</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>31</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8054</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.6088</td>
<td style="text-align:right;">0.0083</td>
<td style="text-align:right;">0.4829</td>
<td style="text-align:right;">0.0162</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>32</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8060</td>
<td style="text-align:right;">0.0228</td>
<td style="text-align:right;">0.6081</td>
<td style="text-align:right;">0.0153</td>
<td style="text-align:right;">0.4816</td>
<td style="text-align:right;">0.0344</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>33</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8066</td>
<td style="text-align:right;">0.0085</td>
<td style="text-align:right;">0.6077</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.4808</td>
<td style="text-align:right;">0.0299</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>34</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8092</td>
<td style="text-align:right;">0.0152</td>
<td style="text-align:right;">0.5973</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.4784</td>
<td style="text-align:right;">0.0076</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>35</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8100</td>
<td style="text-align:right;">0.0147</td>
<td style="text-align:right;">0.5962</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.4769</td>
<td style="text-align:right;">0.0223</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>36</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8125</td>
<td style="text-align:right;">0.0074</td>
<td style="text-align:right;">0.5875</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.4734</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>37</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8125</td>
<td style="text-align:right;">0.0074</td>
<td style="text-align:right;">0.5875</td>
<td style="text-align:right;">0.0100</td>
<td style="text-align:right;">0.4734</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>38</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8140</td>
<td style="text-align:right;">0.0083</td>
<td style="text-align:right;">0.6124</td>
<td style="text-align:right;">0.0086</td>
<td style="text-align:right;">0.4716</td>
<td style="text-align:right;">0.0214</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>39</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8146</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.6174</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.4707</td>
<td style="text-align:right;">0.0242</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>40</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8154</td>
<td style="text-align:right;">0.0180</td>
<td style="text-align:right;">0.6170</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.4704</td>
<td style="text-align:right;">0.0072</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>41</strong></th>
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
<th scope="row" style="text-align:left;"><strong>42</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8158</td>
<td style="text-align:right;">0.0044</td>
<td style="text-align:right;">0.6189</td>
<td style="text-align:right;">0.0080</td>
<td style="text-align:right;">0.4691</td>
<td style="text-align:right;">0.0256</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>43</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8193</td>
<td style="text-align:right;">0.0065</td>
<td style="text-align:right;">0.5918</td>
<td style="text-align:right;">0.0096</td>
<td style="text-align:right;">0.4645</td>
<td style="text-align:right;">0.0261</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>44</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8197</td>
<td style="text-align:right;">0.0124</td>
<td style="text-align:right;">0.6029</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.4642</td>
<td style="text-align:right;">0.0258</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>45</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8197</td>
<td style="text-align:right;">0.0124</td>
<td style="text-align:right;">0.6029</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.4642</td>
<td style="text-align:right;">0.0258</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>46</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8202</td>
<td style="text-align:right;">0.0157</td>
<td style="text-align:right;">0.6057</td>
<td style="text-align:right;">0.0148</td>
<td style="text-align:right;">0.4636</td>
<td style="text-align:right;">0.0226</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>47</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8209</td>
<td style="text-align:right;">0.0169</td>
<td style="text-align:right;">0.6156</td>
<td style="text-align:right;">0.0112</td>
<td style="text-align:right;">0.4629</td>
<td style="text-align:right;">0.0219</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>48</strong></th>
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
<th scope="row" style="text-align:left;"><strong>49</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8219</td>
<td style="text-align:right;">0.0180</td>
<td style="text-align:right;">0.6207</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.4612</td>
<td style="text-align:right;">0.0296</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>50</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8235</td>
<td style="text-align:right;">0.0154</td>
<td style="text-align:right;">0.6023</td>
<td style="text-align:right;">0.0093</td>
<td style="text-align:right;">0.4594</td>
<td style="text-align:right;">0.0216</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>51</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8237</td>
<td style="text-align:right;">0.0184</td>
<td style="text-align:right;">0.6130</td>
<td style="text-align:right;">0.0159</td>
<td style="text-align:right;">0.4590</td>
<td style="text-align:right;">0.0253</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>52</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8251</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.6297</td>
<td style="text-align:right;">0.0141</td>
<td style="text-align:right;">0.4576</td>
<td style="text-align:right;">0.0153</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>53</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8252</td>
<td style="text-align:right;">0.0141</td>
<td style="text-align:right;">0.6224</td>
<td style="text-align:right;">0.0090</td>
<td style="text-align:right;">0.4572</td>
<td style="text-align:right;">0.0206</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>54</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8258</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.6113</td>
<td style="text-align:right;">0.0169</td>
<td style="text-align:right;">0.4566</td>
<td style="text-align:right;">0.0149</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>55</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8265</td>
<td style="text-align:right;">0.0164</td>
<td style="text-align:right;">0.6236</td>
<td style="text-align:right;">0.0164</td>
<td style="text-align:right;">0.4552</td>
<td style="text-align:right;">0.0270</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>56</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8297</td>
<td style="text-align:right;">0.0126</td>
<td style="text-align:right;">0.6141</td>
<td style="text-align:right;">0.0063</td>
<td style="text-align:right;">0.4510</td>
<td style="text-align:right;">0.0268</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>57</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8298</td>
<td style="text-align:right;">0.0170</td>
<td style="text-align:right;">0.6047</td>
<td style="text-align:right;">0.0089</td>
<td style="text-align:right;">0.4512</td>
<td style="text-align:right;">0.0218</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>58</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8323</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.6335</td>
<td style="text-align:right;">0.0054</td>
<td style="text-align:right;">0.4474</td>
<td style="text-align:right;">0.0262</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>59</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8324</td>
<td style="text-align:right;">0.0134</td>
<td style="text-align:right;">0.6225</td>
<td style="text-align:right;">0.0090</td>
<td style="text-align:right;">0.4474</td>
<td style="text-align:right;">0.0261</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>60</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8326</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.6229</td>
<td style="text-align:right;">0.0093</td>
<td style="text-align:right;">0.4472</td>
<td style="text-align:right;">0.0264</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>61</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8327</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.6133</td>
<td style="text-align:right;">0.0146</td>
<td style="text-align:right;">0.4477</td>
<td style="text-align:right;">0.0095</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>62</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8330</td>
<td style="text-align:right;">0.0074</td>
<td style="text-align:right;">0.6355</td>
<td style="text-align:right;">0.0096</td>
<td style="text-align:right;">0.4466</td>
<td style="text-align:right;">0.0228</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>63</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8336</td>
<td style="text-align:right;">0.0090</td>
<td style="text-align:right;">0.6297</td>
<td style="text-align:right;">0.0088</td>
<td style="text-align:right;">0.4454</td>
<td style="text-align:right;">0.0328</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>64</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8338</td>
<td style="text-align:right;">0.0153</td>
<td style="text-align:right;">0.6305</td>
<td style="text-align:right;">0.0084</td>
<td style="text-align:right;">0.4454</td>
<td style="text-align:right;">0.0286</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>65</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8350</td>
<td style="text-align:right;">0.0050</td>
<td style="text-align:right;">0.6166</td>
<td style="text-align:right;">0.0075</td>
<td style="text-align:right;">0.4439</td>
<td style="text-align:right;">0.0258</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>66</strong></th>
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
<th scope="row" style="text-align:left;"><strong>67</strong></th>
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
<th scope="row" style="text-align:left;"><strong>68</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8380</td>
<td style="text-align:right;">0.0142</td>
<td style="text-align:right;">0.6180</td>
<td style="text-align:right;">0.0142</td>
<td style="text-align:right;">0.4402</td>
<td style="text-align:right;">0.0216</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>69</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8384</td>
<td style="text-align:right;">0.0107</td>
<td style="text-align:right;">0.6289</td>
<td style="text-align:right;">0.0066</td>
<td style="text-align:right;">0.4390</td>
<td style="text-align:right;">0.0317</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>70</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8388</td>
<td style="text-align:right;">0.0092</td>
<td style="text-align:right;">0.6215</td>
<td style="text-align:right;">0.0095</td>
<td style="text-align:right;">0.4392</td>
<td style="text-align:right;">0.0177</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>71</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8428</td>
<td style="text-align:right;">0.0119</td>
<td style="text-align:right;">0.6209</td>
<td style="text-align:right;">0.0092</td>
<td style="text-align:right;">0.4336</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>72</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8434</td>
<td style="text-align:right;">0.0133</td>
<td style="text-align:right;">0.6303</td>
<td style="text-align:right;">0.0116</td>
<td style="text-align:right;">0.4332</td>
<td style="text-align:right;">0.0127</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>73</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8435</td>
<td style="text-align:right;">0.0129</td>
<td style="text-align:right;">0.6178</td>
<td style="text-align:right;">0.0090</td>
<td style="text-align:right;">0.4327</td>
<td style="text-align:right;">0.0242</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>74</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8451</td>
<td style="text-align:right;">0.0136</td>
<td style="text-align:right;">0.6353</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.4308</td>
<td style="text-align:right;">0.0174</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>75</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8457</td>
<td style="text-align:right;">0.0175</td>
<td style="text-align:right;">0.6418</td>
<td style="text-align:right;">0.0140</td>
<td style="text-align:right;">0.4300</td>
<td style="text-align:right;">0.0201</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>76</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8464</td>
<td style="text-align:right;">0.0087</td>
<td style="text-align:right;">0.6269</td>
<td style="text-align:right;">0.0110</td>
<td style="text-align:right;">0.4288</td>
<td style="text-align:right;">0.0213</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>77</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8472</td>
<td style="text-align:right;">0.0092</td>
<td style="text-align:right;">0.6516</td>
<td style="text-align:right;">0.0139</td>
<td style="text-align:right;">0.4278</td>
<td style="text-align:right;">0.0195</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>78</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8486</td>
<td style="text-align:right;">0.0133</td>
<td style="text-align:right;">0.6332</td>
<td style="text-align:right;">0.0115</td>
<td style="text-align:right;">0.4256</td>
<td style="text-align:right;">0.0305</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>79</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8488</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.6466</td>
<td style="text-align:right;">0.0156</td>
<td style="text-align:right;">0.4257</td>
<td style="text-align:right;">0.0209</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>80</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8507</td>
<td style="text-align:right;">0.0098</td>
<td style="text-align:right;">0.6491</td>
<td style="text-align:right;">0.0121</td>
<td style="text-align:right;">0.4229</td>
<td style="text-align:right;">0.0227</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>81</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8530</td>
<td style="text-align:right;">0.0105</td>
<td style="text-align:right;">0.6509</td>
<td style="text-align:right;">0.0089</td>
<td style="text-align:right;">0.4194</td>
<td style="text-align:right;">0.0322</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>82</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8538</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.6480</td>
<td style="text-align:right;">0.0079</td>
<td style="text-align:right;">0.4190</td>
<td style="text-align:right;">0.0177</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>83</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8545</td>
<td style="text-align:right;">0.0073</td>
<td style="text-align:right;">0.6561</td>
<td style="text-align:right;">0.0117</td>
<td style="text-align:right;">0.4177</td>
<td style="text-align:right;">0.0244</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>84</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8559</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.6435</td>
<td style="text-align:right;">0.0116</td>
<td style="text-align:right;">0.4162</td>
<td style="text-align:right;">0.0155</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>85</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8596</td>
<td style="text-align:right;">0.0320</td>
<td style="text-align:right;">0.6546</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.4117</td>
<td style="text-align:right;">0.0208</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>86</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8612</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.6589</td>
<td style="text-align:right;">0.0104</td>
<td style="text-align:right;">0.4085</td>
<td style="text-align:right;">0.0257</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>87</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8613</td>
<td style="text-align:right;">0.0111</td>
<td style="text-align:right;">0.6337</td>
<td style="text-align:right;">0.0108</td>
<td style="text-align:right;">0.4087</td>
<td style="text-align:right;">0.0175</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>88</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8633</td>
<td style="text-align:right;">0.0303</td>
<td style="text-align:right;">0.6630</td>
<td style="text-align:right;">0.0180</td>
<td style="text-align:right;">0.4064</td>
<td style="text-align:right;">0.0240</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>89</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8636</td>
<td style="text-align:right;">0.0064</td>
<td style="text-align:right;">0.6592</td>
<td style="text-align:right;">0.0078</td>
<td style="text-align:right;">0.4051</td>
<td style="text-align:right;">0.0289</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>90</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8649</td>
<td style="text-align:right;">0.0329</td>
<td style="text-align:right;">0.6627</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.4044</td>
<td style="text-align:right;">0.0195</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>91</strong></th>
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
<th scope="row" style="text-align:left;"><strong>92</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8687</td>
<td style="text-align:right;">0.0097</td>
<td style="text-align:right;">0.6650</td>
<td style="text-align:right;">0.0088</td>
<td style="text-align:right;">0.3979</td>
<td style="text-align:right;">0.0318</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>93</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8687</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.6679</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.3990</td>
<td style="text-align:right;">0.0033</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>94</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8701</td>
<td style="text-align:right;">0.0326</td>
<td style="text-align:right;">0.6665</td>
<td style="text-align:right;">0.0206</td>
<td style="text-align:right;">0.3973</td>
<td style="text-align:right;">0.0199</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>95</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8709</td>
<td style="text-align:right;">0.0629</td>
<td style="text-align:right;">0.6228</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.3957</td>
<td style="text-align:right;">0.0605</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>96</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.8714</td>
<td style="text-align:right;">0.0210</td>
<td style="text-align:right;">0.6455</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">0.3952</td>
<td style="text-align:right;">0.0119</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>97</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.8726</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">0.6606</td>
<td style="text-align:right;">0.0158</td>
<td style="text-align:right;">0.3931</td>
<td style="text-align:right;">0.0172</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>98</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8734</td>
<td style="text-align:right;">0.0330</td>
<td style="text-align:right;">0.6753</td>
<td style="text-align:right;">0.0217</td>
<td style="text-align:right;">0.3926</td>
<td style="text-align:right;">0.0212</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>99</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8736</td>
<td style="text-align:right;">0.0337</td>
<td style="text-align:right;">0.6499</td>
<td style="text-align:right;">0.0128</td>
<td style="text-align:right;">0.3924</td>
<td style="text-align:right;">0.0233</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>100</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8773</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.6779</td>
<td style="text-align:right;">0.0178</td>
<td style="text-align:right;">0.3869</td>
<td style="text-align:right;">0.0072</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>101</strong></th>
<td style="text-align:right;">rdkit_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8780</td>
<td style="text-align:right;">0.0112</td>
<td style="text-align:right;">0.6855</td>
<td style="text-align:right;">0.0092</td>
<td style="text-align:right;">0.3851</td>
<td style="text-align:right;">0.0293</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>102</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8781</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.6775</td>
<td style="text-align:right;">0.0191</td>
<td style="text-align:right;">0.3858</td>
<td style="text-align:right;">0.0060</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>103</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8805</td>
<td style="text-align:right;">0.0305</td>
<td style="text-align:right;">0.6795</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.3828</td>
<td style="text-align:right;">0.0160</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>104</strong></th>
<td style="text-align:right;">morgan_r1_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8805</td>
<td style="text-align:right;">0.0292</td>
<td style="text-align:right;">0.6750</td>
<td style="text-align:right;">0.0163</td>
<td style="text-align:right;">0.3826</td>
<td style="text-align:right;">0.0171</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>105</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8819</td>
<td style="text-align:right;">0.0279</td>
<td style="text-align:right;">0.6797</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.3794</td>
<td style="text-align:right;">0.0420</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>106</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8831</td>
<td style="text-align:right;">0.0129</td>
<td style="text-align:right;">0.6712</td>
<td style="text-align:right;">0.0102</td>
<td style="text-align:right;">0.3777</td>
<td style="text-align:right;">0.0344</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>107</strong></th>
<td style="text-align:right;">torsion_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8847</td>
<td style="text-align:right;">0.0150</td>
<td style="text-align:right;">0.6818</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.3760</td>
<td style="text-align:right;">0.0263</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>108</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8863</td>
<td style="text-align:right;">0.0146</td>
<td style="text-align:right;">0.6806</td>
<td style="text-align:right;">0.0169</td>
<td style="text-align:right;">0.3736</td>
<td style="text-align:right;">0.0269</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>109</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8865</td>
<td style="text-align:right;">0.0095</td>
<td style="text-align:right;">0.6944</td>
<td style="text-align:right;">0.0093</td>
<td style="text-align:right;">0.3734</td>
<td style="text-align:right;">0.0251</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>110</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8873</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.6827</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.3722</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>111</strong></th>
<td style="text-align:right;">morgan_r1_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.8878</td>
<td style="text-align:right;">0.0127</td>
<td style="text-align:right;">0.6646</td>
<td style="text-align:right;">0.0076</td>
<td style="text-align:right;">0.3717</td>
<td style="text-align:right;">0.0209</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>112</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8882</td>
<td style="text-align:right;">0.0119</td>
<td style="text-align:right;">0.6904</td>
<td style="text-align:right;">0.0116</td>
<td style="text-align:right;">0.3705</td>
<td style="text-align:right;">0.0359</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>113</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8882</td>
<td style="text-align:right;">0.0161</td>
<td style="text-align:right;">0.6812</td>
<td style="text-align:right;">0.0150</td>
<td style="text-align:right;">0.3707</td>
<td style="text-align:right;">0.0330</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>114</strong></th>
<td style="text-align:right;">morgan_r1_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8903</td>
<td style="text-align:right;">0.0185</td>
<td style="text-align:right;">0.6896</td>
<td style="text-align:right;">0.0166</td>
<td style="text-align:right;">0.3684</td>
<td style="text-align:right;">0.0158</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>115</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.8929</td>
<td style="text-align:right;">0.0118</td>
<td style="text-align:right;">0.6903</td>
<td style="text-align:right;">0.0101</td>
<td style="text-align:right;">0.3643</td>
<td style="text-align:right;">0.0259</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>116</strong></th>
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
<th scope="row" style="text-align:left;"><strong>117</strong></th>
<td style="text-align:right;">morgan_r2_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.8989</td>
<td style="text-align:right;">0.0304</td>
<td style="text-align:right;">0.6934</td>
<td style="text-align:right;">0.0179</td>
<td style="text-align:right;">0.3567</td>
<td style="text-align:right;">0.0154</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>118</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9019</td>
<td style="text-align:right;">0.0171</td>
<td style="text-align:right;">0.6696</td>
<td style="text-align:right;">0.0132</td>
<td style="text-align:right;">0.3509</td>
<td style="text-align:right;">0.0400</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>119</strong></th>
<td style="text-align:right;">atom_pair_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9024</td>
<td style="text-align:right;">0.0344</td>
<td style="text-align:right;">0.7000</td>
<td style="text-align:right;">0.0209</td>
<td style="text-align:right;">0.3514</td>
<td style="text-align:right;">0.0305</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>120</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9051</td>
<td style="text-align:right;">0.0208</td>
<td style="text-align:right;">0.7046</td>
<td style="text-align:right;">0.0198</td>
<td style="text-align:right;">0.3474</td>
<td style="text-align:right;">0.0134</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>121</strong></th>
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
<th scope="row" style="text-align:left;"><strong>122</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9084</td>
<td style="text-align:right;">0.0266</td>
<td style="text-align:right;">0.6905</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.3413</td>
<td style="text-align:right;">0.0482</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>123</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9086</td>
<td style="text-align:right;">0.0205</td>
<td style="text-align:right;">0.6995</td>
<td style="text-align:right;">0.0199</td>
<td style="text-align:right;">0.3422</td>
<td style="text-align:right;">0.0184</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>124</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9100</td>
<td style="text-align:right;">0.0167</td>
<td style="text-align:right;">0.6967</td>
<td style="text-align:right;">0.0166</td>
<td style="text-align:right;">0.3401</td>
<td style="text-align:right;">0.0148</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>125</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9105</td>
<td style="text-align:right;">0.0211</td>
<td style="text-align:right;">0.7036</td>
<td style="text-align:right;">0.0207</td>
<td style="text-align:right;">0.3393</td>
<td style="text-align:right;">0.0241</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>126</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9114</td>
<td style="text-align:right;">0.0506</td>
<td style="text-align:right;">0.6758</td>
<td style="text-align:right;">0.0174</td>
<td style="text-align:right;">0.3388</td>
<td style="text-align:right;">0.0435</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>127</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9114</td>
<td style="text-align:right;">0.0506</td>
<td style="text-align:right;">0.6758</td>
<td style="text-align:right;">0.0174</td>
<td style="text-align:right;">0.3388</td>
<td style="text-align:right;">0.0435</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>128</strong></th>
<td style="text-align:right;">morgan_r2_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9125</td>
<td style="text-align:right;">0.0264</td>
<td style="text-align:right;">0.7069</td>
<td style="text-align:right;">0.0177</td>
<td style="text-align:right;">0.3358</td>
<td style="text-align:right;">0.0413</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>129</strong></th>
<td style="text-align:right;">morgan_r1_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9139</td>
<td style="text-align:right;">0.0828</td>
<td style="text-align:right;">0.6488</td>
<td style="text-align:right;">0.0204</td>
<td style="text-align:right;">0.3340</td>
<td style="text-align:right;">0.0899</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>130</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9163</td>
<td style="text-align:right;">0.0330</td>
<td style="text-align:right;">0.7120</td>
<td style="text-align:right;">0.0202</td>
<td style="text-align:right;">0.3317</td>
<td style="text-align:right;">0.0159</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>131</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9183</td>
<td style="text-align:right;">0.0470</td>
<td style="text-align:right;">0.7021</td>
<td style="text-align:right;">0.0148</td>
<td style="text-align:right;">0.3288</td>
<td style="text-align:right;">0.0377</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>132</strong></th>
<td style="text-align:right;">morgan_r2_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9186</td>
<td style="text-align:right;">0.0190</td>
<td style="text-align:right;">0.7165</td>
<td style="text-align:right;">0.0165</td>
<td style="text-align:right;">0.3277</td>
<td style="text-align:right;">0.0171</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>133</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9199</td>
<td style="text-align:right;">0.0188</td>
<td style="text-align:right;">0.7070</td>
<td style="text-align:right;">0.0144</td>
<td style="text-align:right;">0.3249</td>
<td style="text-align:right;">0.0375</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>134</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9225</td>
<td style="text-align:right;">0.0238</td>
<td style="text-align:right;">0.7028</td>
<td style="text-align:right;">0.0172</td>
<td style="text-align:right;">0.3216</td>
<td style="text-align:right;">0.0336</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>135</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9225</td>
<td style="text-align:right;">0.0238</td>
<td style="text-align:right;">0.7029</td>
<td style="text-align:right;">0.0172</td>
<td style="text-align:right;">0.3215</td>
<td style="text-align:right;">0.0336</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>136</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9265</td>
<td style="text-align:right;">0.0490</td>
<td style="text-align:right;">0.6871</td>
<td style="text-align:right;">0.0157</td>
<td style="text-align:right;">0.3166</td>
<td style="text-align:right;">0.0412</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>137</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9273</td>
<td style="text-align:right;">0.0449</td>
<td style="text-align:right;">0.6911</td>
<td style="text-align:right;">0.0175</td>
<td style="text-align:right;">0.3155</td>
<td style="text-align:right;">0.0352</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>138</strong></th>
<td style="text-align:right;">morgan_r2_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9285</td>
<td style="text-align:right;">0.0418</td>
<td style="text-align:right;">0.6967</td>
<td style="text-align:right;">0.0109</td>
<td style="text-align:right;">0.3137</td>
<td style="text-align:right;">0.0323</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>139</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9289</td>
<td style="text-align:right;">0.0279</td>
<td style="text-align:right;">0.6851</td>
<td style="text-align:right;">0.0229</td>
<td style="text-align:right;">0.3123</td>
<td style="text-align:right;">0.0351</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>140</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9289</td>
<td style="text-align:right;">0.0279</td>
<td style="text-align:right;">0.6851</td>
<td style="text-align:right;">0.0229</td>
<td style="text-align:right;">0.3123</td>
<td style="text-align:right;">0.0351</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>141</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9321</td>
<td style="text-align:right;">0.0217</td>
<td style="text-align:right;">0.7075</td>
<td style="text-align:right;">0.0114</td>
<td style="text-align:right;">0.3071</td>
<td style="text-align:right;">0.0366</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>142</strong></th>
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
<th scope="row" style="text-align:left;"><strong>143</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">hgb</td>
<td style="text-align:right;">0.9361</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.7142</td>
<td style="text-align:right;">0.0160</td>
<td style="text-align:right;">0.3013</td>
<td style="text-align:right;">0.0344</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>144</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">svr</td>
<td style="text-align:right;">0.9384</td>
<td style="text-align:right;">0.0294</td>
<td style="text-align:right;">0.6917</td>
<td style="text-align:right;">0.0244</td>
<td style="text-align:right;">0.2979</td>
<td style="text-align:right;">0.0397</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>145</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9408</td>
<td style="text-align:right;">0.0309</td>
<td style="text-align:right;">0.7336</td>
<td style="text-align:right;">0.0218</td>
<td style="text-align:right;">0.2954</td>
<td style="text-align:right;">0.0145</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>146</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9414</td>
<td style="text-align:right;">0.0162</td>
<td style="text-align:right;">0.7224</td>
<td style="text-align:right;">0.0161</td>
<td style="text-align:right;">0.2936</td>
<td style="text-align:right;">0.0252</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>147</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9434</td>
<td style="text-align:right;">0.0212</td>
<td style="text-align:right;">0.7442</td>
<td style="text-align:right;">0.0217</td>
<td style="text-align:right;">0.2911</td>
<td style="text-align:right;">0.0126</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>148</strong></th>
<td style="text-align:right;">torsion_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9440</td>
<td style="text-align:right;">0.0215</td>
<td style="text-align:right;">0.7454</td>
<td style="text-align:right;">0.0215</td>
<td style="text-align:right;">0.2903</td>
<td style="text-align:right;">0.0052</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>149</strong></th>
<td style="text-align:right;">morgan_r0_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9441</td>
<td style="text-align:right;">0.0364</td>
<td style="text-align:right;">0.7337</td>
<td style="text-align:right;">0.0190</td>
<td style="text-align:right;">0.2904</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>150</strong></th>
<td style="text-align:right;">morgan_r0_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9441</td>
<td style="text-align:right;">0.0364</td>
<td style="text-align:right;">0.7337</td>
<td style="text-align:right;">0.0190</td>
<td style="text-align:right;">0.2904</td>
<td style="text-align:right;">0.0225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>151</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9465</td>
<td style="text-align:right;">0.0252</td>
<td style="text-align:right;">0.7374</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.2861</td>
<td style="text-align:right;">0.0288</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>152</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9465</td>
<td style="text-align:right;">0.0252</td>
<td style="text-align:right;">0.7374</td>
<td style="text-align:right;">0.0200</td>
<td style="text-align:right;">0.2861</td>
<td style="text-align:right;">0.0288</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>153</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9472</td>
<td style="text-align:right;">0.0197</td>
<td style="text-align:right;">0.7483</td>
<td style="text-align:right;">0.0195</td>
<td style="text-align:right;">0.2853</td>
<td style="text-align:right;">0.0104</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>154</strong></th>
<td style="text-align:right;">morgan_r1_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9473</td>
<td style="text-align:right;">0.0293</td>
<td style="text-align:right;">0.6983</td>
<td style="text-align:right;">0.0151</td>
<td style="text-align:right;">0.2852</td>
<td style="text-align:right;">0.0230</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>155</strong></th>
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
<th scope="row" style="text-align:left;"><strong>156</strong></th>
<td style="text-align:right;">rdkit_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9519</td>
<td style="text-align:right;">0.0315</td>
<td style="text-align:right;">0.7489</td>
<td style="text-align:right;">0.0226</td>
<td style="text-align:right;">0.2787</td>
<td style="text-align:right;">0.0154</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>157</strong></th>
<td style="text-align:right;">morgan_r0_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9557</td>
<td style="text-align:right;">0.0339</td>
<td style="text-align:right;">0.7441</td>
<td style="text-align:right;">0.0185</td>
<td style="text-align:right;">0.2729</td>
<td style="text-align:right;">0.0184</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>158</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9563</td>
<td style="text-align:right;">0.0216</td>
<td style="text-align:right;">0.7564</td>
<td style="text-align:right;">0.0203</td>
<td style="text-align:right;">0.2716</td>
<td style="text-align:right;">0.0112</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>159</strong></th>
<td style="text-align:right;">rdkit_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9584</td>
<td style="text-align:right;">0.0184</td>
<td style="text-align:right;">0.7376</td>
<td style="text-align:right;">0.0137</td>
<td style="text-align:right;">0.2673</td>
<td style="text-align:right;">0.0398</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>160</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9599</td>
<td style="text-align:right;">0.0257</td>
<td style="text-align:right;">0.7388</td>
<td style="text-align:right;">0.0227</td>
<td style="text-align:right;">0.2661</td>
<td style="text-align:right;">0.0154</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>161</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9611</td>
<td style="text-align:right;">0.0236</td>
<td style="text-align:right;">0.7518</td>
<td style="text-align:right;">0.0196</td>
<td style="text-align:right;">0.2636</td>
<td style="text-align:right;">0.0348</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>162</strong></th>
<td style="text-align:right;">atom_pair_bits_512</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9675</td>
<td style="text-align:right;">0.0265</td>
<td style="text-align:right;">0.7369</td>
<td style="text-align:right;">0.0212</td>
<td style="text-align:right;">0.2545</td>
<td style="text-align:right;">0.0204</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>163</strong></th>
<td style="text-align:right;">torsion_count_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9707</td>
<td style="text-align:right;">0.0265</td>
<td style="text-align:right;">0.7687</td>
<td style="text-align:right;">0.0219</td>
<td style="text-align:right;">0.2498</td>
<td style="text-align:right;">0.0084</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>164</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9744</td>
<td style="text-align:right;">0.0238</td>
<td style="text-align:right;">0.7682</td>
<td style="text-align:right;">0.0210</td>
<td style="text-align:right;">0.2437</td>
<td style="text-align:right;">0.0113</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>165</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9744</td>
<td style="text-align:right;">0.0238</td>
<td style="text-align:right;">0.7682</td>
<td style="text-align:right;">0.0210</td>
<td style="text-align:right;">0.2437</td>
<td style="text-align:right;">0.0113</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>166</strong></th>
<td style="text-align:right;">morgan_r0_bits_2048</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9755</td>
<td style="text-align:right;">0.0182</td>
<td style="text-align:right;">0.7448</td>
<td style="text-align:right;">0.0135</td>
<td style="text-align:right;">0.2417</td>
<td style="text-align:right;">0.0191</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>167</strong></th>
<td style="text-align:right;">morgan_r0_bits_1024</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9762</td>
<td style="text-align:right;">0.0181</td>
<td style="text-align:right;">0.7454</td>
<td style="text-align:right;">0.0134</td>
<td style="text-align:right;">0.2406</td>
<td style="text-align:right;">0.0199</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>168</strong></th>
<td style="text-align:right;">rdkit_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9770</td>
<td style="text-align:right;">0.0560</td>
<td style="text-align:right;">0.7425</td>
<td style="text-align:right;">0.0166</td>
<td style="text-align:right;">0.2401</td>
<td style="text-align:right;">0.0523</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>169</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">elasticnet</td>
<td style="text-align:right;">0.9867</td>
<td style="text-align:right;">0.0207</td>
<td style="text-align:right;">0.7805</td>
<td style="text-align:right;">0.0198</td>
<td style="text-align:right;">0.2244</td>
<td style="text-align:right;">0.0169</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>170</strong></th>
<td style="text-align:right;">torsion_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9884</td>
<td style="text-align:right;">0.0720</td>
<td style="text-align:right;">0.7266</td>
<td style="text-align:right;">0.0264</td>
<td style="text-align:right;">0.2212</td>
<td style="text-align:right;">0.0833</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>171</strong></th>
<td style="text-align:right;">rdkit_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">0.9915</td>
<td style="text-align:right;">0.0069</td>
<td style="text-align:right;">0.7662</td>
<td style="text-align:right;">0.0091</td>
<td style="text-align:right;">0.2155</td>
<td style="text-align:right;">0.0415</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>172</strong></th>
<td style="text-align:right;">morgan_r0_bits_512</td>
<td style="text-align:right;">rf</td>
<td style="text-align:right;">0.9950</td>
<td style="text-align:right;">0.0144</td>
<td style="text-align:right;">0.7625</td>
<td style="text-align:right;">0.0103</td>
<td style="text-align:right;">0.2110</td>
<td style="text-align:right;">0.0211</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>173</strong></th>
<td style="text-align:right;">morgan_r1_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0153</td>
<td style="text-align:right;">0.1073</td>
<td style="text-align:right;">0.6914</td>
<td style="text-align:right;">0.0275</td>
<td style="text-align:right;">0.1766</td>
<td style="text-align:right;">0.1363</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>174</strong></th>
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
<th scope="row" style="text-align:left;"><strong>175</strong></th>
<td style="text-align:right;">atom_pair_bits_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0434</td>
<td style="text-align:right;">0.0427</td>
<td style="text-align:right;">0.7574</td>
<td style="text-align:right;">0.0264</td>
<td style="text-align:right;">0.1297</td>
<td style="text-align:right;">0.0870</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>176</strong></th>
<td style="text-align:right;">atom_pair_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.0885</td>
<td style="text-align:right;">0.2271</td>
<td style="text-align:right;">0.6948</td>
<td style="text-align:right;">0.0345</td>
<td style="text-align:right;">0.0262</td>
<td style="text-align:right;">0.3803</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>177</strong></th>
<td style="text-align:right;">torsion_count_1024</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.1323</td>
<td style="text-align:right;">0.1029</td>
<td style="text-align:right;">0.7352</td>
<td style="text-align:right;">0.0130</td>
<td style="text-align:right;">-0.0389</td>
<td style="text-align:right;">0.2321</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>178</strong></th>
<td style="text-align:right;">morgan_r2_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.1858</td>
<td style="text-align:right;">0.0247</td>
<td style="text-align:right;">0.9211</td>
<td style="text-align:right;">0.0129</td>
<td style="text-align:right;">-0.1204</td>
<td style="text-align:right;">0.0301</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>179</strong></th>
<td style="text-align:right;">morgan_r2_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2050</td>
<td style="text-align:right;">0.0737</td>
<td style="text-align:right;">0.9070</td>
<td style="text-align:right;">0.0192</td>
<td style="text-align:right;">-0.1560</td>
<td style="text-align:right;">0.0887</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>180</strong></th>
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
<th scope="row" style="text-align:left;"><strong>181</strong></th>
<td style="text-align:right;">atom_pair_bits_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.2625</td>
<td style="text-align:right;">0.0884</td>
<td style="text-align:right;">0.8920</td>
<td style="text-align:right;">0.0219</td>
<td style="text-align:right;">-0.2705</td>
<td style="text-align:right;">0.1310</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>182</strong></th>
<td style="text-align:right;">rdkit_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.3146</td>
<td style="text-align:right;">0.0976</td>
<td style="text-align:right;">0.9738</td>
<td style="text-align:right;">0.0474</td>
<td style="text-align:right;">-0.3766</td>
<td style="text-align:right;">0.1398</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>183</strong></th>
<td style="text-align:right;">atom_pair_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">1.8767</td>
<td style="text-align:right;">0.3474</td>
<td style="text-align:right;">1.0119</td>
<td style="text-align:right;">0.0661</td>
<td style="text-align:right;">-1.8707</td>
<td style="text-align:right;">0.9225</td>
</tr>
<tr>
<th scope="row" style="text-align:left;"><strong>184</strong></th>
<td style="text-align:right;">torsion_count_2048</td>
<td style="text-align:right;">ridge</td>
<td style="text-align:right;">2.5947</td>
<td style="text-align:right;">2.0814</td>
<td style="text-align:right;">0.9467</td>
<td style="text-align:right;">0.0986</td>
<td style="text-align:right;">-7.8710</td>
<td style="text-align:right;">13.8215</td>
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
