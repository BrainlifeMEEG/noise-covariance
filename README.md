# noise-covariance

Brainlife App to compute noise covariance matrix from MEG/EEG data using MNE-Python's [mne.compute_covariance](https://mne.tools/stable/generated/mne.compute_covariance.html) function.

## Description

This app estimates the noise covariance matrix, which captures the statistical structure of sensor noise. The noise covariance is essential for source reconstruction — it allows the inverse operator to properly weight channels and separate brain signals from noise.

The app supports three input types:
- **Epochs** (recommended): Uses the pre-stimulus baseline period to estimate noise
- **Empty-room recording**: Uses an entire empty-room raw recording (best noise estimate)
- **Evoked**: Falls back to a simple diagonal covariance (least accurate)

## Inputs

- **epochs** (MNE Epochs FIF): Epoched MEG/EEG data. The pre-stimulus baseline (before time 0) is used to estimate noise covariance.
- **empty_room** (MNE Raw FIF, optional): Empty-room recording. If provided, this takes priority over epochs for noise estimation.

## Outputs

- **noise-cov.fif**: Noise covariance matrix in MNE format (used by the inverse operator app)
- **product.json**: Brainlife report with quality diagnostics and visualizations
- **eigenvalue_spectrum.png**: Eigenvalue spectrum of the covariance matrix
- **noise_covariance.png**: Covariance matrix visualization

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **method** | string | `shrunk` | Covariance estimator. Options: `shrunk` (regularized, recommended), `empirical` (simple), `auto` (tries shrunk then empirical) |
| **tmax** | float | `0.0` | Upper time limit (seconds) for the baseline window. `0.0` means use only pre-stimulus data. |
| **rank** | string | `auto` | Rank estimation for the covariance. `auto` lets MNE determine the rank automatically. Set to an integer to override. |

### Parameter Guidance

- **method**: Use `shrunk` if you have enough data (>5 samples per channel). Use `empirical` if shrunk fails. Use `auto` to try both.
- **tmax**: Keep at `0.0` for event-related designs (uses only pre-stimulus baseline). Set to a positive value only if you want to include post-stimulus data in the noise estimate.
- **rank**: Leave as `auto` unless you know the data rank (e.g., after Maxwell filtering or ICA, the rank is reduced).

## Usage

Configuration file example:
```json
{
    "epochs": "/path/to/epochs-epo.fif",
    "empty_room": "",
    "method": "shrunk",
    "tmax": 0.0,
    "rank": "auto"
}
```

## Quality Diagnostics

The app reports:
- **Effective rank**: How many independent dimensions the covariance captures
- **Condition number**: Ratio of largest to smallest eigenvalue (>1e12 warns of poor estimation)
- **Samples/channel ratio**: Whether there's enough data for reliable estimation (warns if <5)
- **Eigenvalue spectrum plot**: Visual check for rank and noise structure

## Pipeline Position

This app is step 2 of the source reconstruction pipeline:

```
[Forward Model] --> [Noise Covariance] --> [Inverse Operator] --> [Source Estimate]
```

## Authors
- [Kami Salibayeva](https://github.com/KSalibay)

## Funding Acknowledgement

brainlife.io is publicly funded and for the sustainability of the project it is helpful to acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

## Citations

1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
2. Gramfort, A., Luessi, M., Larson, E., et al. MEG and EEG data analysis with MNE-Python. Front. Neurosci. 7, 267 (2013). [https://doi.org/10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
