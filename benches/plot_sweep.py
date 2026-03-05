#!/usr/bin/env python3
"""Plot zenresize performance sweep: throughput vs input size, by ratio and feature mode."""

import csv
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['in_size'] = int(row['in_size'])
            row['in_pixels'] = int(row['in_pixels'])
            row['out_pixels'] = int(row['out_pixels'])
            row['mean_ms'] = float(row['mean_ms'])
            row['stddev_ms'] = float(row['stddev_ms'])
            row['in_mpps'] = float(row['in_mpps'])
            row['out_mpps'] = float(row['out_mpps'])
            rows.append(row)
    return rows

def main():
    safe_data = load_csv('/tmp/sweep_safe.csv')
    unsafe_data = load_csv('/tmp/sweep_unsafe.csv')

    import os
    outdir = os.environ.get('SWEEP_OUTPUT_DIR', '/mnt/v/output/zenresize/sweep')
    os.makedirs(outdir, exist_ok=True)

    # --- Chart 1: Throughput (input MP/s) vs input size, one line per ratio ---
    # Safe mode only, downscale ratios
    fig, ax = plt.subplots(figsize=(14, 8))

    ratio_order = ['12.5%', '25%', '33%', '50%', '67%', '75%', '100%']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ratio_order)))

    for i, ratio in enumerate(ratio_order):
        pts = sorted([r for r in safe_data if r['ratio'] == ratio], key=lambda r: r['in_size'])
        if not pts:
            continue
        sizes = [p['in_size'] for p in pts]
        mpps = [p['in_mpps'] for p in pts]
        ax.plot(sizes, mpps, 'o-', color=colors[i], label=f'→{ratio}', linewidth=2, markersize=5)

    ax.set_xlabel('Input size (square, pixels per side)', fontsize=12)
    ax.set_ylabel('Throughput (input MP/s)', fontsize=12)
    ax.set_title('zenresize sRGB throughput vs input size (safe mode, downscale)', fontsize=14)
    ax.legend(title='Output ratio', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    fig.tight_layout()
    fig.savefig(f'{outdir}/throughput_vs_size_safe.png', dpi=150)
    print(f'Saved {outdir}/throughput_vs_size_safe.png')
    plt.close()

    # --- Chart 2: Safe vs Unsafe throughput comparison at 50% downscale ---
    fig, ax = plt.subplots(figsize=(14, 8))

    for label, data, color, marker in [
        ('safe', safe_data, '#1f77b4', 'o'),
        ('unsafe_kernels', unsafe_data, '#ff7f0e', 's'),
    ]:
        pts = sorted([r for r in data if r['ratio'] == '50%'], key=lambda r: r['in_size'])
        sizes = [p['in_size'] for p in pts]
        mpps = [p['in_mpps'] for p in pts]
        ax.plot(sizes, mpps, f'{marker}-', color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Input size (square, pixels per side)', fontsize=12)
    ax.set_ylabel('Throughput (input MP/s)', fontsize=12)
    ax.set_title('zenresize sRGB: safe vs unsafe_kernels (50% downscale)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    fig.tight_layout()
    fig.savefig(f'{outdir}/safe_vs_unsafe_50pct.png', dpi=150)
    print(f'Saved {outdir}/safe_vs_unsafe_50pct.png')
    plt.close()

    # --- Chart 3: Throughput vs ratio at fixed sizes ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    target_sizes = [1024, 2048, 3840, 7680]

    ratio_to_float = {
        '12.5%': 0.125, '25%': 0.25, '33%': 0.333, '50%': 0.5,
        '67%': 0.667, '75%': 0.75, '100%': 1.0, '150%': 1.5,
        '200%': 2.0, '300%': 3.0,
    }

    for ax, size in zip(axes.flat, target_sizes):
        for label, data, color, marker in [
            ('safe', safe_data, '#1f77b4', 'o'),
            ('unsafe_kernels', unsafe_data, '#ff7f0e', 's'),
        ]:
            pts = sorted(
                [r for r in data if r['in_size'] == size],
                key=lambda r: ratio_to_float.get(r['ratio'], 0)
            )
            if not pts:
                continue
            ratios = [ratio_to_float[p['ratio']] for p in pts]
            mpps = [p['in_mpps'] for p in pts]
            ax.plot(ratios, mpps, f'{marker}-', color=color, label=label, linewidth=2, markersize=6)

        ax.set_xlabel('Scale ratio', fontsize=10)
        ax.set_ylabel('Throughput (input MP/s)', fontsize=10)
        ax.set_title(f'{size}x{size}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('zenresize sRGB throughput vs scale ratio', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{outdir}/throughput_vs_ratio.png', dpi=150)
    print(f'Saved {outdir}/throughput_vs_ratio.png')
    plt.close()

    # --- Chart 4: Overhead heatmap (safe time / unsafe time) ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Build lookup for unsafe times
    unsafe_lookup = {}
    for r in unsafe_data:
        unsafe_lookup[(r['in_size'], r['ratio'])] = r['mean_ms']

    all_sizes = sorted(set(r['in_size'] for r in safe_data))
    all_ratios_str = ['12.5%', '25%', '33%', '50%', '67%', '75%', '100%', '150%', '200%', '300%']

    overhead_matrix = []
    valid_sizes = []
    valid_ratios = []

    for ratio in all_ratios_str:
        row = []
        for size in all_sizes:
            safe_ms = None
            unsafe_ms = None
            for r in safe_data:
                if r['in_size'] == size and r['ratio'] == ratio:
                    safe_ms = r['mean_ms']
                    break
            unsafe_ms = unsafe_lookup.get((size, ratio))
            if safe_ms is not None and unsafe_ms is not None and unsafe_ms > 0:
                row.append((safe_ms / unsafe_ms - 1.0) * 100.0)  # percent overhead
            else:
                row.append(float('nan'))
        overhead_matrix.append(row)

    overhead_arr = np.array(overhead_matrix)

    im = ax.imshow(overhead_arr, aspect='auto', cmap='RdYlGn_r', vmin=-5, vmax=25,
                   interpolation='nearest')
    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(all_ratios_str)))
    ax.set_yticklabels(all_ratios_str, fontsize=10)
    ax.set_xlabel('Input size (square)', fontsize=12)
    ax.set_ylabel('Scale ratio', fontsize=12)
    ax.set_title('Bounds-checking overhead: (safe / unsafe_kernels − 1) × 100%', fontsize=14)

    # Annotate cells
    for i in range(len(all_ratios_str)):
        for j in range(len(all_sizes)):
            val = overhead_arr[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 15 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label='Overhead %')
    fig.tight_layout()
    fig.savefig(f'{outdir}/overhead_heatmap.png', dpi=150)
    print(f'Saved {outdir}/overhead_heatmap.png')
    plt.close()

    # --- Chart 5: Time vs input megapixels (log-log), all ratios ---
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, ratio in enumerate(ratio_order):
        pts = sorted([r for r in safe_data if r['ratio'] == ratio], key=lambda r: r['in_pixels'])
        if not pts:
            continue
        mpx = [p['in_pixels'] / 1e6 for p in pts]
        ms = [p['mean_ms'] for p in pts]
        ax.plot(mpx, ms, 'o-', color=colors[i], label=f'→{ratio}', linewidth=2, markersize=5)

    ax.set_xlabel('Input megapixels', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('zenresize sRGB time vs input size (safe mode)', fontsize=14)
    ax.legend(title='Output ratio', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(f'{outdir}/time_vs_mpx_loglog.png', dpi=150)
    print(f'Saved {outdir}/time_vs_mpx_loglog.png')
    plt.close()

    print('\nDone. All charts saved to', outdir)

if __name__ == '__main__':
    main()
