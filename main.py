"""
app-noise-covariance: Compute noise covariance matrix for source reconstruction.

Brainlife.io app that computes noise covariance from epochs baseline or
empty-room recordings. Outputs noise-cov.fif for use by the inverse operator app.

Inputs: Epochs (baseline used for covariance) or empty-room raw recording.
Outputs: noise-cov.fif, MNE HTML report with covariance plots.
"""

# Copyright (c) 2026 brainlife.io
#
# Authors:
# - Maximilien Chaumon (https://github.com/dnacombo)
# - obVdo (https://github.com/obVdo)

import os
import sys
import json
import base64

# When deployed on Brainlife: brainlife_utils/ and source_recon_utils.py are in this directory.
# When running locally in the monorepo: they're in the parent directory.
_app_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_app_dir)

# Try local first (Brainlife deployment), then parent (local monorepo dev)
for _path in [_app_dir, _parent_dir]:
    if os.path.isdir(os.path.join(_path, 'brainlife_utils')):
        sys.path.insert(0, _path)
        break

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source_recon_utils import (
    load_input_data, compute_noise_covariance, save_outputs,
)


def load_config():
    """Load config.json and clean up Brainlife metadata keys."""
    with open('config.json') as f:
        config = json.load(f)

    # Convert "" to None
    for k, v in config.items():
        if v == "":
            config[k] = None

    # Strip Brainlife internal keys
    for key in ['_app', '_tid', '_inputs', '_outputs', '_rule']:
        config.pop(key, None)

    return config


def main():
    # Clean and make output dirs
    import shutil
    for d in ['out_dir', 'out_figs', 'out_dir_report']:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    config = load_config()

    # ad_hoc_fallback: if False (default), the app fails when no epochs or empty-room
    # are provided. If True, falls back to a diagonal (ad-hoc) covariance when only
    # evoked data is available. Expose as a checkbox in the Brainlife UI.
    ad_hoc_fallback = str(config.get('ad_hoc_fallback', 'false')).lower() in ('true', '1', 'yes')

    def _fail(msg):
        """Write error product.json and exit with code 1 (marks app as failed on Brainlife)."""
        print(f"ERROR: {msg}")
        error_product = {'brainlife': [{'type': 'error', 'msg': msg}]}
        with open('product.json', 'w') as f:
            json.dump(error_product, f)
        sys.exit(1)

    # == STEP 1: Load input data ==
    # Priority: empty_room > epochs > raw > evoked
    empty_room_file = config.get('empty_room')
    data = None

    if empty_room_file and os.path.exists(empty_room_file):
        data = mne.io.read_raw_fif(empty_room_file, preload=True)
        print(f"Loaded empty-room recording: {len(data.ch_names)} channels")

    if data is None:
        try:
            data = load_input_data(config)
        except (FileNotFoundError, Exception) as e:
            # No input at all is always an error — ad_hoc_fallback requires at least evoked data.
            _fail(
                f"No valid input provided. Provide 'epochs' or 'empty_room' in config.json. "
                f"Checked: evoked='{config.get('evoked')}', epochs='{config.get('epochs')}', "
                f"raw='{config.get('raw')}'"
            )
        if isinstance(data, mne.Evoked) and not ad_hoc_fallback:
            _fail(
                "Only evoked data provided — cannot compute proper noise covariance from evoked. "
                "Provide 'epochs' or 'empty_room', or enable 'ad_hoc_fallback' for a diagonal estimate."
            )
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

    # Save bad channels list (empty file if none found)
    bad_ch_file = os.path.join('out_dir', 'bad_channels.txt')
    with open(bad_ch_file, 'w') as f:
        for ch in all_new_bads:
            f.write(ch + '\n')
    if all_new_bads:
        print(f"Bad channels written to {bad_ch_file}")

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

    import time
    t0 = time.time()
    noise_cov = compute_noise_covariance(data, tmin=tmin, tmax=tmax, method=method, rank=rank)
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
    from collections import Counter
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

    # Plot 0: Channel variance with bad channel detection (if auto_bad was run)
    if ch_variance_info:
        n_types = len(ch_variance_info)
        fig_var, axes_var = plt.subplots(n_types, 1, figsize=(14, 4 * n_types), squeeze=False)
        bad_set = set(all_new_bads)
        for ax_idx, (ch_type, (_, variances, med, thresh)) in enumerate(ch_variance_info.items()):
            ax = axes_var[ax_idx, 0]
            colors = ['red' if v > thresh * med or (med > 0 and v < med / 100) else 'steelblue'
                       for v in variances]
            ax.bar(range(len(variances)), variances, width=1.0, color=colors)
            if med > 0:
                ax.axhline(med, color='green', ls='--', lw=1, label='Median')
                ax.axhline(thresh * med, color='orange', ls='--', lw=1, label=f'{thresh:.0f}x median')
            ax.set_xlabel('Channel index')
            ax.set_ylabel('Variance (log scale)')
            ax.set_yscale('log')
            n_bad = sum(1 for c in colors if c == 'red')
            ax.set_title(f'{ch_type.upper()} channel variance ({n_bad} bad in red)')
            ax.legend(loc='upper right')

            # Inset: MNE-native sensor topomap with bad channels marked
            try:
                axins = ax.inset_axes([0.78, 0.52, 0.20, 0.44])
                # Temporarily mark detected bads so MNE highlights them
                info_tmp = data.info.copy()
                info_tmp['bads'] = list(bad_set)
                mne.viz.plot_sensors(info_tmp, ch_type=ch_type, axes=axins,
                                     show=False, show_names=False)
                axins.set_title('Sensors', fontsize=7, pad=2)
            except Exception as e:
                print(f"Could not draw sensor inset for {ch_type}: {e}")

        plt.suptitle('Auto Bad Channel Detection', fontsize=14, y=1.01)
        plt.tight_layout()
        var_path = os.path.join('out_figs', 'channel_variance.png')
        fig_var.savefig(var_path, dpi=150, bbox_inches='tight')
        plt.close(fig_var)
        report.add_image(var_path, title='Channel Variance (Bad Channel Detection)')

    # Plot 1: Covariance matrix + channel noise spectra (mne.viz.plot_cov)
    try:
        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info, show=False)

        cov_path = os.path.join('out_figs', 'noise_covariance.png')
        fig_cov.savefig(cov_path, dpi=150, bbox_inches='tight')
        plt.close(fig_cov)
        report.add_image(cov_path, title='Covariance Matrix')

        spectra_path = os.path.join('out_figs', 'noise_spectra.png')
        fig_spectra.savefig(spectra_path, dpi=150, bbox_inches='tight')
        plt.close(fig_spectra)
        report.add_image(spectra_path, title='Channel Noise Spectra')
    except Exception as e:
        print(f"Could not plot covariance: {e}")

    # Plot 2: Whitened evoked (evoked.plot_white) — only if we have epochs
    if isinstance(data, mne.BaseEpochs):
        try:
            evoked = data.average()
            fig_white = evoked.plot_white(noise_cov, show=False)
            white_path = os.path.join('out_figs', 'whitened_evoked.png')
            fig_white.savefig(white_path, dpi=150, bbox_inches='tight')
            plt.close(fig_white)
            report.add_image(white_path, title='Whitened Evoked (GFP)')
        except Exception as e:
            print(f"Could not plot whitened evoked: {e}")

    # Plot 3: Covariance topomaps — only if we have epochs
    if isinstance(data, mne.BaseEpochs):
        try:
            evoked = data.average()
            fig_topo = noise_cov.plot_topomap(evoked.info, show=False)
            topo_path = os.path.join('out_figs', 'covariance_topomaps.png')
            fig_topo.savefig(topo_path, dpi=150, bbox_inches='tight')
            plt.close(fig_topo)
            report.add_image(topo_path, title='Noise Covariance Topomaps')
        except Exception as e:
            print(f"Could not plot topomaps: {e}")

    # Save HTML report
    report_path = os.path.join('out_dir_report', 'report.html')
    report.save(report_path, overwrite=True)
    print(f"Report saved to {report_path}")

    # == STEP 4: Save outputs ==
    save_outputs(noise_cov=noise_cov, out_dir='out_dir')

    # == STEP 5: product.json with info + figure thumbnails ==
    dict_json_product = {'brainlife': []}

    # Add text info messages
    for msg in report_msgs:
        msg_type = 'warning' if msg.startswith('WARNING') else 'info'
        dict_json_product['brainlife'].append({
            'type': msg_type,
            'msg': msg,
        })

    for img_name, img_path in [
        ('Channel Variance', 'out_figs/channel_variance.png'),
        ('Covariance Matrix', 'out_figs/noise_covariance.png'),
        ('Channel Noise Spectra', 'out_figs/noise_spectra.png'),
        ('Whitened Evoked', 'out_figs/whitened_evoked.png'),
        ('Covariance Topomaps', 'out_figs/covariance_topomaps.png'),
    ]:
        if os.path.exists(img_path):
            data_uri = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
            dict_json_product['brainlife'].append({
                'type': 'image/png',
                'name': img_name,
                'base64': data_uri,
            })

    with open('product.json', 'w') as f:
        json.dump(dict_json_product, f)

    print("Done.")


if __name__ == '__main__':
    main()
