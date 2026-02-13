"""
Helper functions for app-noise-covariance.

Contains the full diagnostic report functions (detect_problems + generate_html_report)
that were removed from main.py to keep it minimal. These can be imported if you want
the detailed diagnostics version.
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt


def detect_problems(data, noise_cov, info, method_used, tmax, fell_back_to_adhoc,
                    fell_back_to_empirical):
    """Detect potential problems and return structured diagnostics.

    Returns
    -------
    problems : list of dict
        Each dict has keys: severity, title, detail, suggestion.
        severity is one of: 'critical', 'warning', 'info', 'good'.
    cov_stats : dict
        eigenvalues, condition_number, effective_rank, n_channels, ch_types.
    """
    problems = []

    # --- Input data checks ---
    if isinstance(data, mne.Evoked):
        problems.append({
            'severity': 'critical',
            'title': 'Ad-hoc covariance from Evoked data',
            'detail': (
                'Input is Evoked (averaged) data, which has no single-trial noise information. '
                'A diagonal ad-hoc covariance was used instead of a real noise estimate.'
            ),
            'suggestion': (
                'Provide Epochs data (pre-averaged) so the baseline period can be used for '
                'noise estimation. Alternatively, provide an empty-room recording.'
            ),
        })

    if fell_back_to_adhoc:
        problems.append({
            'severity': 'critical',
            'title': 'Fell back to ad-hoc diagonal covariance',
            'detail': 'The covariance computation failed and fell back to mne.make_ad_hoc_cov.',
            'suggestion': 'Provide more epochs, use method="empirical", or check your input data.',
        })

    if fell_back_to_empirical:
        problems.append({
            'severity': 'info',
            'title': 'Fell back to empirical method (scikit-learn not available)',
            'detail': 'The "shrunk" estimator requires scikit-learn. Used empirical instead.',
            'suggestion': 'This is usually fine with enough data (samples/channel > 5).',
        })

    # --- Epoch-specific checks ---
    if isinstance(data, mne.BaseEpochs):
        n_epochs = len(data)
        n_channels = len(mne.pick_types(data.info, meg=True, eeg=True, exclude='bads'))
        n_baseline_samples = int(np.sum(data.times <= tmax))
        total_samples = n_epochs * n_baseline_samples
        ratio = total_samples / max(n_channels, 1)

        if n_epochs < 10:
            problems.append({
                'severity': 'warning',
                'title': f'Very few epochs ({n_epochs})',
                'detail': f'Only {n_epochs} epochs. Consider 30-50+ for reliable estimation.',
                'suggestion': 'Add more epochs or use an empty-room recording.',
            })

        if ratio < 3:
            problems.append({
                'severity': 'critical',
                'title': f'Critically low samples/channel ratio ({ratio:.1f})',
                'detail': f'{total_samples} baseline samples for {n_channels} channels.',
                'suggestion': 'More epochs, extend baseline, use empty-room, or method="empirical".',
            })
        elif ratio < 5:
            problems.append({
                'severity': 'warning',
                'title': f'Low samples/channel ratio ({ratio:.1f})',
                'detail': f'{total_samples} baseline samples for {n_channels} channels.',
                'suggestion': 'A ratio above 5 is recommended.',
            })
        else:
            problems.append({
                'severity': 'good',
                'title': f'Adequate samples/channel ratio ({ratio:.1f})',
                'detail': f'{total_samples} baseline samples for {n_channels} channels.',
                'suggestion': '',
            })

        if tmax > 0:
            problems.append({
                'severity': 'warning',
                'title': f'Baseline includes post-stimulus (tmax={tmax}s)',
                'detail': 'Post-stimulus data included in noise estimate.',
                'suggestion': 'Set tmax=0.0 for event-related designs.',
            })

        if data.tmin >= 0:
            problems.append({
                'severity': 'critical',
                'title': 'No pre-stimulus baseline in epochs',
                'detail': f'Epochs start at tmin={data.tmin}s.',
                'suggestion': 'Re-epoch with negative tmin or use empty-room recording.',
            })

    # --- EEG reference check ---
    has_eeg = len(mne.pick_types(info, eeg=True, exclude='bads')) > 0
    if has_eeg:
        has_avg_ref = any(
            'average' in p['desc'].lower() and 'eeg' in p['desc'].lower()
            for p in info['projs']
        )
        if not has_avg_ref:
            problems.append({
                'severity': 'warning',
                'title': 'No average EEG reference projection',
                'detail': 'EEG data without average reference can affect covariance quality.',
                'suggestion': 'Add average EEG reference in preprocessing.',
            })

    # --- Covariance quality checks ---
    cov_data = noise_cov.data
    if hasattr(cov_data, 'toarray'):
        cov_data = cov_data.toarray()

    eigenvalues = np.linalg.eigvalsh(cov_data)
    eigenvalues = np.sort(eigenvalues)[::-1]

    n_total = len(eigenvalues)
    pos_eigs = eigenvalues[eigenvalues > 0]
    neg_eigs = eigenvalues[eigenvalues < 0]

    if len(neg_eigs) > 0:
        problems.append({
            'severity': 'warning',
            'title': f'Negative eigenvalues detected ({len(neg_eigs)})',
            'detail': f'{len(neg_eigs)} of {n_total} eigenvalues are negative.',
            'suggestion': 'MNE handles this by regularization. Add more data if possible.',
        })

    if len(pos_eigs) > 0:
        condition_number = pos_eigs[0] / pos_eigs[-1]
    else:
        condition_number = float('inf')

    threshold = eigenvalues[0] * 1e-10
    effective_rank = int(np.sum(eigenvalues > threshold))

    ch_types_present = []
    n_meg = len(mne.pick_types(info, meg=True, exclude='bads'))
    n_eeg = len(mne.pick_types(info, eeg=True, exclude='bads'))
    if n_meg > 0:
        ch_types_present.append(f'{n_meg} MEG')
    if n_eeg > 0:
        ch_types_present.append(f'{n_eeg} EEG')

    if n_meg > 0 and n_eeg > 0:
        problems.append({
            'severity': 'info',
            'title': f'Mixed channel types: {", ".join(ch_types_present)}',
            'detail': 'High condition number expected with mixed MEG+EEG.',
            'suggestion': '',
        })
    elif condition_number > 1e12:
        problems.append({
            'severity': 'warning',
            'title': f'Very high condition number ({condition_number:.1e})',
            'detail': 'Large eigenvalue ratio suggests poor estimation.',
            'suggestion': 'Check bad channels, add data, or use method="shrunk".',
        })

    n_projs = sum(1 for p in info['projs'] if p['active'])
    if n_projs > 0 and effective_rank < n_total:
        problems.append({
            'severity': 'info',
            'title': f'Rank reduced by {n_total - effective_rank} ({n_projs} active projectors)',
            'detail': f'Effective rank: {effective_rank}/{n_total}. Normal with SSP.',
            'suggestion': '',
        })

    n_bads = len(info['bads'])
    if n_bads > 0:
        problems.append({
            'severity': 'info',
            'title': f'{n_bads} bad channel(s) excluded',
            'detail': f'Bad channels: {", ".join(info["bads"])}',
            'suggestion': '',
        })

    return problems, {
        'eigenvalues': eigenvalues,
        'condition_number': condition_number,
        'effective_rank': effective_rank,
        'n_channels': n_total,
        'ch_types': ch_types_present,
    }


def generate_html_report(data, noise_cov, info, problems, cov_stats,
                         method_used, tmax):
    """Generate a comprehensive MNE HTML report with diagnostics and FAQ.

    This is the full diagnostic report from app-noise-covariance v1.
    Use this if you want detailed text warnings and contextual help.
    """
    report = mne.Report(title='Noise Covariance Report')

    # Summary table
    if isinstance(data, mne.BaseEpochs):
        data_summary = (
            f'{len(data)} epochs, {len(data.ch_names)} channels, '
            f'baseline [{data.tmin:.3f}, {tmax:.3f}]s'
        )
    elif isinstance(data, mne.io.BaseRaw):
        data_summary = (
            f'Raw recording, {len(data.ch_names)} channels, '
            f'{data.n_times / data.info["sfreq"]:.1f}s duration'
        )
    else:
        data_summary = f'{type(data).__name__}, {len(data.ch_names)} channels'

    summary_html = f"""
    <div style="font-family: sans-serif; padding: 10px;">
        <h3>Noise Covariance Estimation Summary</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 6px; font-weight: bold;">Input data:</td>
                <td style="padding: 6px;">{data_summary}</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Method:</td>
                <td style="padding: 6px;">{method_used}</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Baseline tmax:</td>
                <td style="padding: 6px;">{tmax}s</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Channel types:</td>
                <td style="padding: 6px;">{', '.join(cov_stats['ch_types'])}</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Effective rank:</td>
                <td style="padding: 6px;">{cov_stats['effective_rank']} / {cov_stats['n_channels']}</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Condition number:</td>
                <td style="padding: 6px;">{cov_stats['condition_number']:.2e}</td></tr>
        </table>
    </div>
    """
    report.add_html(title='Summary', html=summary_html)

    # Problems section
    severity_colors = {
        'critical': '#d32f2f', 'warning': '#f57c00',
        'info': '#1976d2', 'good': '#388e3c',
    }
    severity_icons = {
        'critical': 'CRITICAL', 'warning': 'WARNING',
        'info': 'INFO', 'good': 'OK',
    }

    if problems:
        problems_html = '<div style="font-family: sans-serif; padding: 10px;">'
        for p in problems:
            color = severity_colors[p['severity']]
            icon = severity_icons[p['severity']]
            bg = {'critical': '#ffebee', 'warning': '#fff3e0',
                   'good': '#e8f5e9'}.get(p['severity'], '#e3f2fd')
            problems_html += f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background: {bg};">
                <strong style="color: {color};">[{icon}] {p['title']}</strong>
                <p style="margin: 5px 0;">{p['detail']}</p>
            """
            if p['suggestion']:
                problems_html += f"""
                <p style="margin: 5px 0; padding: 8px; background: #f5f5f5; border-radius: 4px;">
                    <strong>What to do:</strong> {p['suggestion'].replace(chr(10), '<br>')}
                </p>
                """
            problems_html += '</div>'
        problems_html += '</div>'
        report.add_html(title='Quality Checks & Troubleshooting', html=problems_html)

    # Eigenvalue spectrum
    eigenvalues = cov_stats['eigenvalues']
    threshold = eigenvalues[0] * 1e-10

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.semilogy(range(1, len(eigenvalues) + 1),
                np.maximum(eigenvalues, 1e-30),
                'o-', markersize=3, linewidth=1, color='#1976D2')
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
               label=f'Rank threshold (rank={cov_stats["effective_rank"]})')
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Covariance Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    eig_path = os.path.join('out_figs', 'eigenvalue_spectrum.png')
    plt.savefig(eig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    report.add_image(eig_path, title='Eigenvalue Spectrum')

    # Covariance matrix + spectra
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
        report.add_html(title='Covariance Plot',
                        html=f'<p>Could not generate: {e}</p>')

    # Topomaps (epochs only)
    if isinstance(data, mne.BaseEpochs):
        try:
            evoked = data.average()
            fig_white = noise_cov.plot_topomap(evoked.info, show=False)
            white_path = os.path.join('out_figs', 'whitened_topomaps.png')
            fig_white.savefig(white_path, dpi=150, bbox_inches='tight')
            plt.close(fig_white)
            report.add_image(white_path, title='Noise Covariance Topomaps')
        except Exception:
            pass

    # Save
    report_path = os.path.join('out_dir_report', 'report.html')
    report.save(report_path, overwrite=True)
    return eig_path
