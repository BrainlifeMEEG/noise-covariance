# Compute Noise Covariance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Estimates the MEG/EEG noise covariance matrix with MNE-Python from an empty-room recording or the pre-stimulus baseline of epochs (or an ad hoc diagonal covariance from evoked data), with diagnostic plots and a report.**

Uses the first available input in priority order: an empty-room recording (mne.compute_raw_covariance), evoked data (diagonal ad hoc covariance, only if ad_hoc_fallback is true), or epochs (mne.compute_covariance on the tmin-tmax baseline). Optionally flags and interpolates channels with outlying variance first. Writes cov.fif, diagnostic plots and an HTML report with rank and conditioning messages. Use it before building an inverse operator; prefer empty-room data or a pre-stimulus baseline over the ad hoc fallback. The ica input is not read by main.py.

| | |
|---|---|
| 📥 **Inputs** | `neuro/meeg/mne/raw` (optional) · `neuro/meeg/mne/epochs` (optional) · `neuro/meeg/mne/evoked` (optional) · `neuro/meeg/mne/ica` (optional) |
| 📤 **Outputs** | `neuro/meeg/mne/covariance` · `report/html` |
| 🐳 **Container** | `brainlifemeeg/mne:1.12.1` |
| ⚙️ **Resources** | 1 CPUs · 32 GB memory · 2 h walltime |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.891](https://doi.org/10.25663/brainlife.app.891) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `empty-room` | `neuro/meeg/mne/raw` | no | Empty-room recording used for noise covariance. This data should be preprocessed (e.g., Maxwell filtered, band-passed) to match the experimental data and remove line noise. |
| `epochs` | `neuro/meeg/mne/epochs` | no | Epoched MEG/EEG data; the pre-stimulus baseline (tmin to tmax) is used to estimate noise covariance. |
| `evoked` | `neuro/meeg/mne/evoked` | no | Use only if epoch or empty room data isn't available. |
| `ica` | `neuro/meeg/mne/ica` | no | ICA solution (MNE); not used by main.py in this version. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `neuro/meeg/mne/covariance` | Computed noise covariance matrix |
| `out_report` | `report/html` | HTML report with covariance matrix, noise spectra and (for epochs) whitening diagnostics. |

<details>
<summary><b>⚙️ Parameters</b> (7)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | enum |  | Estimator method for noise covariance. Options: `shrunk`, `empirical`, `diagonal_fixed`, `auto`. |
| `rank` | string | `None` | Rank estimation parameter ('info', 'full', or a 'dict'[eg: {'mag': 90, 'eeg': 45}]). |
| `tmin` | string | `None` | Time corresponding to first sample of data used for covariance calculation. None or a number |
| `tmax` | string | `0` | Time corresponding to last sample of data used for covariance calculation. None or a number |
| `auto_bad_channels` | boolean | `false` | Automatically detect and interpolate bad channels based on variance. |
| `bad_channel_threshold` | number | `5` | Variance threshold multiplier. Channels with variance > N times the median will be marked bad (e.g., 5). |
| `ad_hoc_fallback` | boolean | `false` | Return ad hoc noise covariance, use only when there is no epochs or empty room recordings |

</details>

<!-- brainlife-app:end -->

[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.891-blue.svg)](https://doi.org/10.25663/brainlife.app.891)

## Description

This app estimates the noise covariance matrix used for MEG/EEG source reconstruction. Depending on which input is supplied, it calls `mne.compute_raw_covariance` (empty-room recording or raw), `mne.compute_covariance` (epochs, using the pre-stimulus baseline), or `mne.make_ad_hoc_cov` (evoked, diagonal ad-hoc fallback). Channels with abnormal baseline variance can optionally be auto-detected and interpolated before the covariance is computed.

The noise covariance is essential for source reconstruction: it lets the inverse operator properly weight channels and separate brain signal from sensor noise.

The app generates:
- A noise covariance file (`cov.fif`)
- Diagnostic plots (channel variance, covariance matrix, noise spectra, and — for epochs input — whitened evoked and covariance topomaps)
- An HTML report bundling those plots
- A `product.json` with quality-diagnostic messages and plot thumbnails

## Inputs

- **`empty_room`** (`neuro/meeg/mne/raw`): Empty-room recording. If present, it takes priority over every other input — the covariance is computed directly from it via `mne.compute_raw_covariance`. Should be preprocessed (e.g. Maxwell-filtered, band-passed) to match the experimental data (optional).
- **`evoked`** (`neuro/meeg/mne/evoked`): Evoked data. If no `empty_room` is given, this is tried next, but only to build a diagonal ad-hoc covariance via `mne.make_ad_hoc_cov`, and only when `ad_hoc_fallback` is `true` — otherwise the app exits with an error (optional; use only if neither epochs nor an empty-room recording is available).
- **`epochs`** (`neuro/meeg/mne/epochs`): Epoched data. Used, if no `empty_room`/`evoked` is given, to estimate the noise covariance from the pre-stimulus baseline via `mne.compute_covariance` (recommended input) (optional).
- **`raw`** (`neuro/meeg/mne/raw`): Continuous recording with events, used as a last resort when none of the above is given. Epochs are created internally from the events found in the recording (using the `tmin`/`tmax` config values) before computing the covariance the same way as for `epochs`. Not currently exposed as an input on brainlife.io's web form; settable via a config override (also accepted under the legacy key `mne`) (optional).

## Outputs

- **`out_dir/cov.fif`** (`neuro/meeg/mne/covariance`): Estimated noise covariance matrix, consumed by the inverse-operator app.
- **`out_figs/*.png`**: Diagnostic plots — channel variance (only if `auto_bad_channels` flagged any), covariance matrix, channel noise spectra, and (epochs input only) whitened evoked GFP and covariance topomaps.
- **`out_report/report.html`** (`report/html`): MNE HTML report bundling the plots above.
- **`product.json`**: Brainlife.io metadata — quality-diagnostic messages (effective rank, condition number, samples/channel, etc.) and base64 thumbnails of the plots.

## Configuration Parameters

| key | type | default | description |
|---|---|---|---|
| `method` | string (enum: `shrunk`, `empirical`, `diagonal_fixed`, `auto`) | `shrunk` | Covariance estimator passed to `mne.compute_covariance`/`mne.compute_raw_covariance`. `auto` tries `shrunk` then falls back to `empirical`. |
| `tmin` | string | `""` (→ `None`) | Lower time bound (seconds) for the baseline/analysis window. Empty or `"None"` = epoch start (epochs input) / recording start (raw/empty-room input). |
| `tmax` | string | `""` (→ `None`) | Upper time bound (seconds) for the baseline/analysis window. Empty or `"None"` = epoch end (epochs input) / recording end (raw/empty-room input). |
| `rank` | string | `""` (→ `None`, auto-detect) | Rank estimation. `None`/empty = auto-detect from data, `info` = from measurement info (e.g. after Maxwell filtering), `full` = assume full rank, or a JSON dict, e.g. `{"eeg": 45}`. |
| `auto_bad_channels` | boolean | `false` | Automatically flag channels (per type: EEG/grad/mag) whose baseline variance is far from the per-type median and interpolate them before computing the covariance. |
| `bad_channel_threshold` | number | `5` | Multiplier on the per-type median baseline variance above which a channel is flagged bad (channels below `median/100` are also flagged as dead). Used only when `auto_bad_channels` is `true`. |
| `ad_hoc_fallback` | boolean | `false` | If `true`, allows falling back to a diagonal ad-hoc covariance (`mne.make_ad_hoc_cov`) when only evoked data is available. If `false` (default), that case is a fatal error. |

### tmin / tmax Behavior

All parameters are strings on Brainlife. `"None"` or empty = Python `None` (MNE default). Any number = that time in seconds.

| | Epochs (`compute_covariance`) | Raw / empty-room (`compute_raw_covariance`) |
|---|---|---|
| **tmin=None** | first sample of epoch | 0 (MNE default) |
| **tmax=None** | last sample of epoch | end of recording |
| **tmin/tmax=float** | that time in seconds | that time in seconds |

### Parameter Guidance

- **method**: Use `shrunk` if you have enough data (>5 samples per channel). Use `empirical` if `shrunk` fails. Use `auto` to try both.
- **tmin/tmax**: For event-related designs, use `tmin=None, tmax=0.0` to use only the pre-stimulus baseline. Leave both as `None`/empty to use the full epoch/recording.
- **rank**: Leave as `None`/empty unless you know the data rank (e.g. after Maxwell filtering or ICA, the rank is reduced). Use `info` if Maxwell filtering was applied.

## Usage

### Running on Brainlife.io

1. Open the app page on brainlife.io ([bl.app.891](https://doi.org/10.25663/brainlife.app.891)).
2. Select an Epochs object as input (recommended), or an empty-room Raw recording for the most accurate estimate; Evoked is an optional fallback input.
3. Set `method`, `tmin`/`tmax`, `rank`, and the bad-channel options as needed — the defaults work for most datasets.
4. Submit the app. Outputs appear as `out_dir/cov.fif`, with the quality-diagnostics report in `out_report/report.html` once the process finishes.

### Local Testing

Example `config.json`:
```json
{
    "epochs": "/path/to/epochs-epo.fif",
    "empty_room": "",
    "evoked": "",
    "raw": "",
    "method": "shrunk",
    "tmin": "",
    "tmax": "0.0",
    "rank": "",
    "auto_bad_channels": false,
    "bad_channel_threshold": 5,
    "ad_hoc_fallback": false
}
```

Run:
```bash
singularity exec docker://brainlifemeeg/mne:1.12.1 python3 main.py
```

## Technical Details

The app reports:
- **Effective rank**: How many independent dimensions the covariance captures
- **Condition number**: Ratio of largest to smallest eigenvalue (>1e12 warns of poor estimation)
- **Samples/channel ratio**: Whether there's enough data for reliable estimation (warns if <5)
- **Eigenvalue spectrum / covariance plots**: Visual check for rank and noise structure

## Pipeline Position

This app is step 2 of the source reconstruction pipeline:

```
[Forward Model] --> [Noise Covariance] --> [Inverse Operator] --> [Source Estimate]
```

## Authors
- [Maximilien Chaumon](https://github.com/dnacombo)
- [obVdo](https://github.com/obVdo)

## Citations

1. Hayashi, S., Caron, B.A., Heinsfeld, A.S. et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. Nat Methods 21, 809–813 (2024). https://doi.org/10.1038/s41592-024-02237-2
2. Gramfort, A., Luessi, M., Larson, E., et al. MEG and EEG data analysis with MNE-Python. Front. Neurosci. 7, 267 (2013). https://doi.org/10.3389/fnins.2013.00267
3. Engemann, D.A., Gramfort, A. Automated model selection in covariance estimation and spatial whitening of MEG and EEG signals. NeuroImage 108, 328–342 (2015). https://doi.org/10.1016/j.neuroimage.2014.12.040

## Funding Acknowledgement

brainlife.io is publicly funded and for the sustainability of the project it is helpful to acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://grantome.com/grant/NIH/R01-EB030896-01)

## License

Copyright (c) 2026 MEEG Brainlife team. Licensed under AGPL-3.0, see [license.txt](license.txt).
