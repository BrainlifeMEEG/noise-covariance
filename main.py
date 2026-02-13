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
# - Kami Salibayeva (https://github.com/KSalibay)

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
            print(f"ERROR: {e}")
            # Write minimal product.json with error info
            error_product = {'brainlife': [{'type': 'error', 'msg': str(e)}]}
            with open('product.json', 'w') as f:
                json.dump(error_product, f)
            return
        if isinstance(data, mne.BaseEpochs):
            print(f"Loaded {len(data)} epochs, {len(data.ch_names)} channels")
        elif isinstance(data, mne.io.BaseRaw):
            print(f"Loaded raw: {len(data.ch_names)} channels")
        else:
            print(f"Loaded {type(data).__name__}")

    # == STEP 2: Compute noise covariance ==
    tmax_val = config.get('tmax')
    tmax = float(tmax_val) if tmax_val is not None else 0.0

    method_str = config.get('method') or 'shrunk'
    if method_str == 'auto':
        method = ['shrunk', 'empirical']
    else:
        method = [method_str]

    rank = config.get('rank') or 'auto'
    if rank == 'auto':
        rank = None
    elif isinstance(rank, str) and rank.isdigit():
        rank = int(rank)

    noise_cov = compute_noise_covariance(data, tmax=tmax, method=method, rank=rank)
    print("Noise covariance computed")

    # == STEP 3: Generate plots and report ==
    info = data.info
    report = mne.Report(title='Noise Covariance Report')

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

    # == STEP 5: product.json with figure thumbnails ==
    dict_json_product = {'brainlife': []}

    for img_name, img_path in [
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
