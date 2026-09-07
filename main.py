"""
Noise covariance app for Brainlife.io

Computes the noise covariance matrix used for source reconstruction, from
epochs baseline, an empty-room recording, or (as a diagonal ad-hoc fallback)
evoked data.

Inputs:
    - epochs (MNE Epochs FIF): pre-stimulus baseline used to estimate noise.
    - empty_room (MNE Raw FIF, optional): empty-room recording; takes
      priority over epochs/evoked/raw when provided.
    - evoked (MNE Evoked FIF, optional): used only for the diagonal ad-hoc
      covariance fallback (enable via 'ad_hoc_fallback').

Outputs:
    - out_dir/cov.fif: Noise covariance matrix in MNE format (used by the inverse operator app)
    - out_figs/*.png: Covariance matrix, noise spectra, and diagnostic plots
    - out_report/report.html: MNE HTML report with covariance plots
    - product.json: Brainlife.io report with quality diagnostics and visualizations
"""

# Copyright (c) 2026 brainlife.io
#
# Authors:
# - Maximilien Chaumon (https://github.com/dnacombo)
# - obVdo (https://github.com/obVdo)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

# Standard imports
import time
import json
from collections import Counter
import numpy as np
import mne
import matplotlib.pyplot as plt

# Import shared utilities
from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
    save_figure_with_base64,
)

# Set up matplotlib for headless execution
setup_matplotlib_backend()

# Ensure output directories exist
ensure_output_dirs('out_dir', 'out_figs', 'out_report')

# Load configuration
config = load_config()

# ad_hoc_fallback: if False (default), the app fails when no epochs or empty-room
# are provided. If True, falls back to a diagonal (ad-hoc) covariance when only
# evoked data is available. Expose as a checkbox in the Brainlife UI.
ad_hoc_fallback = str(config.get('ad_hoc_fallback', 'false')).lower() in ('true', '1', 'yes')

# Accumulator for product.json (see "Product Metadata Convention")
product_items = []

# == STEP 1: Load input data ==
# Priority: empty_room > epochs > raw > evoked

# Resolve config paths that may point to a Brainlife directory (when no File
# Mapping is configured) rather than a file, for the 'evoked'/'epochs'/'raw'
# inputs. Default filenames per datatype are tried first, falling back to
# the single .fif file in the directory if there's exactly one.
_default_filenames = {
    'evoked': ['evoked-ave.fif', 'meg-ave.fif'],
    'epochs': ['meg-epo.fif', 'epochs-epo.fif'],
    'raw': ['raw.fif'],
}
_resolved = {}
for _key, _cfg_path in [
    ('evoked', config.get('evoked', '')),
    ('epochs', config.get('epochs', '')),
    ('raw', config.get('raw', '') or config.get('mne', '')),
]:
    _resolved_path = _cfg_path
    if _cfg_path and os.path.isdir(_cfg_path):
        _resolved_path = None
        for _fname in _default_filenames.get(_key, []):
            _candidate = os.path.join(_cfg_path, _fname)
            if os.path.isfile(_candidate):
                print(f"Resolved {_key}: {_cfg_path} -> {_candidate}")
                _resolved_path = _candidate
                break
        if _resolved_path is None:
            _fif_files = [f for f in os.listdir(_cfg_path) if f.endswith('.fif')]
            if len(_fif_files) == 1:
                _resolved_path = os.path.join(_cfg_path, _fif_files[0])
                print(f"Resolved {_key}: {_cfg_path} -> {_resolved_path}")
            elif len(_fif_files) > 1:
                print(f"Warning: multiple .fif files in {_cfg_path}: {_fif_files}. Using first match.")
                _resolved_path = os.path.join(_cfg_path, _fif_files[0])
            else:
                print(f"Warning: no .fif file found in directory {_cfg_path}")
                _resolved_path = None
    _resolved[_key] = _resolved_path

evoked_file = _resolved['evoked']
epochs_file = _resolved['epochs']
raw_file = _resolved['raw']

empty_room_file = config.get('empty_room')
data = None

if empty_room_file and os.path.exists(empty_room_file):
    data = mne.io.read_raw_fif(empty_room_file, preload=True)
    print(f"Loaded empty-room recording: {len(data.ch_names)} channels")

if data is None:
    try:
        # Try evoked first
        if evoked_file and os.path.exists(evoked_file):
            evoked = mne.read_evokeds(evoked_file)
            if isinstance(evoked, list):
                evoked = evoked[0]
            print(f"Loaded evoked data from {evoked_file}: {evoked.nave} averages, "
                  f"{len(evoked.ch_names)} channels")
            data = evoked

        # Try epochs
        elif epochs_file and os.path.exists(epochs_file):
            if epochs_file.endswith('.bdf'):
                _bdf_raw = mne.io.read_raw_bdf(epochs_file, preload=True)
                _bdf_events = mne.find_events(_bdf_raw)
                _ep_tmin = float(config.get('tmin', -0.2))
                _ep_tmax = float(config.get('tmax', 0.5))
                epochs = mne.Epochs(_bdf_raw, _bdf_events, tmin=_ep_tmin, tmax=_ep_tmax, preload=True)
            else:
                epochs = mne.read_epochs(epochs_file, preload=True)
            print(f"Loaded {len(epochs)} epochs from {epochs_file}, "
                  f"{len(epochs.ch_names)} channels")
            # Add EEG average reference projector if missing (required for inverse)
            if 'eeg' in epochs:
                _has_ref = any(proj['desc'] == 'Average EEG reference'
                               for proj in epochs.info['projs'])
                if not _has_ref:
                    print("Adding missing EEG average reference projector")
                    epochs.set_eeg_reference('average', projection=True)
                    epochs.apply_proj()
            data = epochs

        # Try raw
        elif raw_file and os.path.exists(raw_file):
            _raw = mne.io.read_raw_fif(raw_file, preload=True)
            _events = mne.find_events(_raw)
            _ep_tmin = float(config.get('tmin', -0.2))
            _ep_tmax = float(config.get('tmax', 0.5))
            epochs = mne.Epochs(_raw, _events, tmin=_ep_tmin, tmax=_ep_tmax, preload=True)
            print(f"Created {len(epochs)} epochs from raw {raw_file}")
            # Add EEG average reference projector if missing (required for inverse)
            if 'eeg' in epochs:
                _has_ref = any(proj['desc'] == 'Average EEG reference'
                               for proj in epochs.info['projs'])
                if not _has_ref:
                    print("Adding missing EEG average reference projector")
                    epochs.set_eeg_reference('average', projection=True)
                    epochs.apply_proj()
            data = epochs

        else:
            raise FileNotFoundError(
                "No valid input file found. Provide 'evoked', 'epochs', or 'raw' "
                f"in config.json. Checked: evoked='{evoked_file}', "
                f"epochs='{epochs_file}', raw='{raw_file}'"
            )
    except Exception as e:
        # No input at all is always an error — ad_hoc_fallback requires at least evoked data.
        _msg = (
            f"No valid input provided. Provide 'epochs' or 'empty_room' in config.json. "
            f"Checked: evoked='{config.get('evoked')}', epochs='{config.get('epochs')}', "
            f"raw='{config.get('raw')}'"
        )
        print(f"ERROR: {_msg}")
        add_info_to_product(product_items, _msg, 'error')
        create_product_json(product_items)
        sys.exit(1)

    if isinstance(data, mne.Evoked) and not ad_hoc_fallback:
        _msg = (
            "Only evoked data provided — cannot compute proper noise covariance from evoked. "
            "Provide 'epochs' or 'empty_room', or enable 'ad_hoc_fallback' for a diagonal estimate."
        )
        print(f"ERROR: {_msg}")
        add_info_to_product(product_items, _msg, 'error')
        create_product_json(product_items)
        sys.exit(1)

    if isinstance(data, mne.BaseEpochs):
        print(f"Loaded {len(data)} epochs, {len(data.ch_names)} channels")
    elif isinstance(data, mne.io.BaseRaw):
        print(f"Loaded raw: {len(data.ch_names)} channels")
    else:
        print(f"Loaded {type(data).__name__} (ad-hoc covariance will be used)")

# == Detect and interpolate bad channels ==
# Flag channels with extreme baseline variance (e.g. noisy or dead channels).
# These crush the covariance colorbar and degrade the estimate.
# Done PER CHANNEL TYPE since MEG grad/mag/EEG have very different scales.
auto_bad_str = str(config.get('auto_bad_channels', 'true')).lower()
auto_bad = auto_bad_str in ('true', '1', 'yes', '')
bad_ch_threshold = float(config.get('bad_channel_threshold', 5))
all_new_bads = []
ch_variance_info = {}  # for plotting: {ch_type: (ch_names, variances, median, threshold)}

if auto_bad and isinstance(data, mne.BaseEpochs):
    existing_bads = list(data.info.get('bads', []))
    for ch_type, pick_kwargs in [
        ('eeg', dict(eeg=True, meg=False)),
        ('grad', dict(meg='grad', eeg=False)),
        ('mag', dict(meg='mag', eeg=False)),
    ]:
        picks = mne.pick_types(data.info, exclude='bads', **pick_kwargs)
        if len(picks) == 0:
            continue
        baseline_data = data.copy().crop(tmin=data.tmin, tmax=0.0).get_data()[:, picks, :]
        ch_var = np.var(baseline_data, axis=(0, 2))
        median_var = np.median(ch_var)
        pick_names = [data.ch_names[p] for p in picks]
        ch_variance_info[ch_type] = (pick_names, ch_var, median_var, bad_ch_threshold)
        if median_var > 0:
            bad_idx = np.where(ch_var > bad_ch_threshold * median_var)[0]
            dead_idx = np.where(ch_var < median_var / 100)[0]
            for idx in np.concatenate([bad_idx, dead_idx]):
                name = data.ch_names[picks[idx]]
                if name not in existing_bads and name not in all_new_bads:
                    all_new_bads.append(name)
    if all_new_bads:
        data.info['bads'] = existing_bads + all_new_bads
        print(f"Auto-detected {len(all_new_bads)} bad channels (>{bad_ch_threshold}x median variance): {all_new_bads}")
        data.interpolate_bads(reset_bads=True)
        print(f"Interpolated {len(all_new_bads)} bad channels")
    else:
        print("No bad channels detected")


# == STEP 2: Compute noise covariance ==
# tmin: baseline start time (seconds). Empty/""/None = use epoch start.
# On Brainlife, leave empty or don't set to use full epoch baseline.
tmin_val = config.get('tmin')
if tmin_val is None or tmin_val == '' or tmin_val == 'None':
    tmin = None  # use epoch start
else:
    tmin = float(tmin_val)

tmax_val = config.get('tmax')
if tmax_val is None or tmax_val == '' or str(tmax_val) == 'None':
    tmax = None  # None = use last sample (MNE default)
else:
    tmax = float(tmax_val)

method_str = config.get('method') or 'shrunk'
if method_str == 'auto':
    method = ['shrunk', 'empirical']
else:
    method = [method_str]

# rank: None | 'info' | 'full' | dict
# None = estimate from data, 'info' = from measurement info/Maxwell,
# 'full' = assume full rank, dict = per-channel-type e.g. {"mag": 90, "eeg": 45}
rank_raw = config.get('rank')
if rank_raw is None or rank_raw == '' or str(rank_raw).lower() in ('none', 'auto'):
    rank = None
elif isinstance(rank_raw, dict):
    # Already a dict from JSON, e.g. {"mag": 90, "eeg": 45}
    rank = {k: int(v) for k, v in rank_raw.items()}
elif isinstance(rank_raw, str) and rank_raw.lower() in ('info', 'full'):
    rank = rank_raw.lower()
elif isinstance(rank_raw, str):
    # Try parsing as JSON dict, e.g. '{"mag": 90, "eeg": 45}'
    try:
        parsed = json.loads(rank_raw)
        if isinstance(parsed, dict):
            rank = {k: int(v) for k, v in parsed.items()}
        else:
            print(f"WARNING: Could not parse rank='{rank_raw}', using None (auto-detect)")
            rank = None
    except (json.JSONDecodeError, ValueError):
        print(f"WARNING: Unknown rank value '{rank_raw}', using None (auto-detect)")
        rank = None
else:
    rank = None

t0 = time.time()

# Evoked -> ad-hoc diagonal covariance
if isinstance(data, mne.Evoked):
    print("Input is Evoked — using ad-hoc diagonal noise covariance")
    noise_cov = mne.make_ad_hoc_cov(data.info)

# Raw (empty-room) -> compute from recording
# mne.compute_raw_covariance defaults: tmin=0, tmax=None (full recording)
elif isinstance(data, mne.io.BaseRaw):
    try:
        raw_kwargs = {}
        if tmin is not None:
            raw_kwargs['tmin'] = tmin
        if tmax is not None:
            raw_kwargs['tmax'] = tmax
        try:
            noise_cov = mne.compute_raw_covariance(data, method=method, rank=rank, verbose=True, **raw_kwargs)
        except ValueError as e:
            # MNE 1.0.2 rejects the full method list if any entry requires
            # sklearn, even if 'empirical' is also listed.
            if 'scikit-learn' in str(e) and method != ['empirical']:
                print("scikit-learn not available, falling back to method='empirical'")
                noise_cov = mne.compute_raw_covariance(data, method=['empirical'], rank=rank, verbose=True, **raw_kwargs)
            else:
                raise
        print("Computed noise covariance from raw recording")
    except Exception as e:
        print(f"Warning: raw covariance failed ({e}), using ad-hoc")
        noise_cov = mne.make_ad_hoc_cov(data.info)

# Epochs -> compute from baseline
else:
    try:
        cov_kwargs = {}
        if tmin is not None:
            cov_kwargs['tmin'] = tmin
        if tmax is not None:
            cov_kwargs['tmax'] = tmax
        try:
            noise_cov = mne.compute_covariance(data, method=method, rank=rank, verbose=True, **cov_kwargs)
        except ValueError as e:
            # MNE 1.0.2 rejects the full method list if any entry requires
            # sklearn, even if 'empirical' is also listed.
            if 'scikit-learn' in str(e) and method != ['empirical']:
                print("scikit-learn not available, falling back to method='empirical'")
                noise_cov = mne.compute_covariance(data, method=['empirical'], rank=rank, verbose=True, **cov_kwargs)
            else:
                raise
        tmin_str = f"{tmin}" if tmin is not None else "epoch start"
        tmax_str = f"{tmax}" if tmax is not None else "epoch end"
        print(f"Computed noise covariance from {len(data)} epochs "
              f"(baseline [{tmin_str}, {tmax_str}]s)")
    except Exception as e:
        print(f"Warning: covariance computation failed ({e}), "
              f"using ad-hoc diagonal covariance")
        noise_cov = mne.make_ad_hoc_cov(data.info)

cov_time = time.time() - t0
print(f"Noise covariance computed in {cov_time:.1f}s")

# == Collect summary info for product.json ==
info = data.info
report_msgs = []

# Input type
if isinstance(data, mne.BaseEpochs):
    input_type = 'epochs'
    actual_tmin = tmin if tmin is not None else data.tmin
    actual_tmax = tmax if tmax is not None else data.tmax
    report_msgs.append(f"Input: {len(data)} epochs, baseline [{actual_tmin:.3f}, {actual_tmax:.3f}] s")
elif isinstance(data, mne.io.BaseRaw):
    input_type = 'raw (empty-room)' if empty_room_file else 'raw'
    report_msgs.append(f"Input: {input_type}, {data.times[-1]:.1f} s duration")
else:
    input_type = type(data).__name__
    report_msgs.append(f"Input: {input_type}")

# Channel summary
ch_types = Counter(mne.channel_type(info, i) for i in range(len(info['ch_names']))
                   if info['ch_names'][i] in noise_cov.ch_names)
ch_summary = ', '.join(f"{v} {k}" for k, v in sorted(ch_types.items()))
report_msgs.append(f"Channels in covariance: {len(noise_cov.ch_names)} ({ch_summary})")

# Bad channel info
if all_new_bads:
    report_msgs.append(f"Auto-detected and interpolated {len(all_new_bads)} bad channels: {', '.join(all_new_bads)}")
elif auto_bad:
    report_msgs.append("Bad channel detection: none found")

# Total baseline duration used for covariance
if isinstance(data, mne.BaseEpochs):
    baseline_per_epoch = actual_tmax - actual_tmin  # seconds per epoch
    total_baseline_sec = len(data) * baseline_per_epoch
    report_msgs.append(
        f"Baseline duration: {baseline_per_epoch:.3f} s/epoch x {len(data)} epochs "
        f"= {total_baseline_sec:.1f} s total"
    )
elif isinstance(data, mne.io.BaseRaw):
    total_baseline_sec = data.times[-1] - data.times[0]
    report_msgs.append(f"Recording duration used: {total_baseline_sec:.1f} s")

# Samples and method
n_samples = noise_cov.nfree + 1
samples_per_ch = n_samples / max(len(noise_cov.ch_names), 1)
total_data_sec = n_samples / info['sfreq']
cov_method = noise_cov.get('method', method_str)
report_msgs.append(f"Method: {cov_method}")
report_msgs.append(f"Samples used: {n_samples} ({samples_per_ch:.1f} per channel, {total_data_sec:.1f} s of data)")
if samples_per_ch < 5:
    report_msgs.append(
        f"WARNING: Low samples/channel ratio ({samples_per_ch:.1f}). "
        "Consider using more data or fewer channels for reliable estimation."
    )

# Rank and condition number
evals = np.linalg.eigvalsh(noise_cov.data)
n_zero = np.sum(evals < evals[-1] * 1e-10)
eff_rank = len(evals) - n_zero
# Skip near-zero eigenvalues (from avg reference, projectors) for condition number
nonzero_evals = evals[evals > evals[-1] * 1e-10]
cond = nonzero_evals[-1] / nonzero_evals[0] if len(nonzero_evals) > 0 else float('inf')
report_msgs.append(f"Effective rank: {eff_rank} / {len(noise_cov.ch_names)}")
if cond > 1e12:
    report_msgs.append(f"Condition number: {cond:.1e} (high -- regularization recommended)")
else:
    report_msgs.append(f"Condition number: {cond:.1e}")

# Projectors
projs = info.get('projs', [])
if projs:
    proj_names = [p['desc'] for p in projs if p['active']]
    report_msgs.append(f"Active projectors ({len(proj_names)}): {', '.join(proj_names)}")

# Sampling frequency
report_msgs.append(f"Sampling frequency: {info['sfreq']:.1f} Hz")

# Compute time
report_msgs.append(f"Computation time: {cov_time:.1f} s")

# == STEP 3: Generate plots and report ==
report = mne.Report(title='Noise Covariance Report')
# Base64 thumbnails for product.json, keyed the same as the img_name/
# img_path pairs used below. Rendered at a much lower dpi than the
# full-res out_figs/ files (via brainlife_utils' save_figure_with_base64,
# the same dual-dpi pattern every other app in this repo uses) -- this
# avoids re-reading the full 150dpi PNG straight off disk and embedding it
# verbatim, which for ~380-channel MEG+EEG data (esp. the covariance
# matrix and whitened-evoked plots) pushed product.json's total size
# past Amaretti's 1MB cap. Confirmed on the ICM cluster.
fig_base64 = {}

# Plot 0: Channel variance with bad channel detection (if auto_bad was run)
if ch_variance_info:
    from matplotlib.patches import Patch
    n_types = len(ch_variance_info)
    fig_var, axes_var = plt.subplots(
        n_types, 2,
        figsize=(16, 4 * n_types),
        gridspec_kw={'width_ratios': [4, 1]},
        squeeze=False,
    )
    bad_set = set(all_new_bads)
    for ax_idx, (ch_type, (_, variances, med, thresh)) in enumerate(ch_variance_info.items()):
        # Left column: channel variance bar chart
        ax = axes_var[ax_idx, 0]
        colors = ['red' if v > thresh * med or (med > 0 and v < med / 100) else 'steelblue'
                   for v in variances]
        ax.bar(range(len(variances)), variances, width=1.0, color=colors)
        if med > 0:
            ax.axhline(med, color='green', ls='--', lw=1, label='Median')
            ax.axhline(thresh * med, color='orange', ls='--', lw=1,
                       label=f'{thresh:.0f}x median (threshold)')
        ax.set_xlabel('Channel index')
        ax.set_ylabel('Variance (log scale)')
        ax.set_yscale('log')
        n_bad = sum(1 for c in colors if c == 'red')
        ax.set_title(f'{ch_type.upper()} channel variance')
        ax.legend(loc='upper right', fontsize=8)

        # Right column: sensor topomap
        ax_topo = axes_var[ax_idx, 1]
        try:
            info_tmp = data.info.copy()
            info_tmp['bads'] = list(bad_set)
            mne.viz.plot_sensors(info_tmp, ch_type=ch_type, axes=ax_topo,
                                 show=False, show_names=False)
            # Equal aspect ratio so head is a circle, not an oval
            ax_topo.set_aspect('equal', adjustable='datalim')
            ax_topo.set_title(f'{ch_type.upper()} sensors\n({n_bad} bad)', fontsize=9)
            legend_els = [
                Patch(facecolor='steelblue', label='Good'),
                Patch(facecolor='red', label=f'Bad ({n_bad})'),
            ]
            ax_topo.legend(handles=legend_els, loc='lower center',
                           fontsize=7, framealpha=0.8)
        except Exception as e:
            ax_topo.axis('off')
            ax_topo.text(0.5, 0.5, f'No topomap\n{e}', ha='center', va='center',
                         transform=ax_topo.transAxes, fontsize=7)
            print(f"Could not draw sensor topomap for {ch_type}: {e}")

    plt.suptitle('Auto Bad Channel Detection', fontsize=14)
    plt.tight_layout()
    var_path = os.path.join('out_figs', 'channel_variance.png')
    fig_base64['Channel Variance'] = save_figure_with_base64(fig_var, var_path)
    report.add_image(var_path, title='Channel Variance (Bad Channel Detection)')

# Plot 1: Covariance matrix + channel noise spectra (mne.viz.plot_cov)
try:
    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info, show=False)

    cov_path = os.path.join('out_figs', 'noise_covariance.png')
    fig_base64['Covariance Matrix'] = save_figure_with_base64(fig_cov, cov_path)
    report.add_image(cov_path, title='Covariance Matrix')

    spectra_path = os.path.join('out_figs', 'noise_spectra.png')
    fig_base64['Channel Noise Spectra'] = save_figure_with_base64(fig_spectra, spectra_path)
    report.add_image(spectra_path, title='Channel Noise Spectra')
except Exception as e:
    print(f"Could not plot covariance: {e}")

# Plot 2: Whitened evoked (evoked.plot_white) — only if we have epochs
if isinstance(data, mne.BaseEpochs):
    try:
        evoked = data.average()
        fig_white = evoked.plot_white(noise_cov, show=False)
        white_path = os.path.join('out_figs', 'whitened_evoked.png')
        fig_base64['Whitened Evoked'] = save_figure_with_base64(fig_white, white_path)
        report.add_image(white_path, title='Whitened Evoked (GFP)')
    except Exception as e:
        print(f"Could not plot whitened evoked: {e}")

# Plot 3: Covariance topomaps — only if we have epochs
if isinstance(data, mne.BaseEpochs):
    try:
        evoked = data.average()
        fig_topo = noise_cov.plot_topomap(evoked.info, show=False)
        topo_path = os.path.join('out_figs', 'covariance_topomaps.png')
        fig_base64['Covariance Topomaps'] = save_figure_with_base64(fig_topo, topo_path)
        report.add_image(topo_path, title='Noise Covariance Topomaps')
    except Exception as e:
        print(f"Could not plot topomaps: {e}")

# Save HTML report
report_path = os.path.join('out_report', 'report.html')
report.save(report_path, overwrite=True)
print(f"Report saved to {report_path}")

# == STEP 4: Save outputs ==
cov_path_out = os.path.join('out_dir', 'cov.fif')
mne.write_cov(cov_path_out, noise_cov, overwrite=True)
print(f"Saved: Noise covariance -> {cov_path_out}")

# == STEP 5: product.json with info + figure thumbnails ==
# Add text info messages
for msg in report_msgs:
    msg_type = 'warning' if msg.startswith('WARNING') else 'info'
    add_info_to_product(product_items, msg, msg_type)

# Use the low-dpi thumbnails computed alongside each figure above (see
# fig_base64), not a raw re-read of the full 150dpi out_figs/ file -- that
# combined easily exceeded Amaretti's 1MB product.json cap.
for img_name, data_uri in fig_base64.items():
    add_image_to_product(product_items, img_name, base64_data=data_uri)

create_product_json(product_items)

print("Done.")
