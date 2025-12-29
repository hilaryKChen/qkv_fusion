#!/usr/bin/env python3
"""
Generate figures for the QKV Fusion Final Report.

Figures:
1. Roofline diagram showing operation placement relative to ridge point
2. Bar chart comparing kernel implementations
3. Pie chart of layer-level latency breakdown

Usage:
    python generate_figures.py
    
Output:
    - fig_roofline.pdf / fig_roofline.png
    - fig_kernel_comparison.pdf / fig_kernel_comparison.png
    - fig_latency_breakdown.pdf / fig_latency_breakdown.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for academic papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fig1_roofline():
    """
    Roofline diagram showing operation placement relative to ridge point.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # A100 specs
    peak_compute = 312.0  # TFLOPS
    peak_bandwidth = 2039  # GB/s
    ridge_point = peak_compute * 1000 / peak_bandwidth  # FLOPs/Byte = 153.0
    
    # Measured values
    measured_compute = 148.4  # TFLOPS
    measured_bandwidth = 1204.9  # GB/s
    measured_ridge = measured_compute * 1000 / measured_bandwidth
    
    # Arithmetic intensity range (log scale)
    ai = np.logspace(0, 3, 1000)  # 1 to 1000 FLOPs/Byte
    
    # Roofline: min(peak_compute, bandwidth * AI)
    # Convert to TFLOPS for plotting
    roofline_theoretical = np.minimum(peak_compute, peak_bandwidth * ai / 1000)
    roofline_measured = np.minimum(measured_compute, measured_bandwidth * ai / 1000)
    
    # Plot rooflines
    ax.loglog(ai, roofline_theoretical, 'b-', linewidth=2, label='Theoretical Peak')
    ax.loglog(ai, roofline_measured, 'b--', linewidth=1.5, label='Measured Peak')
    
    # Ridge point line
    ax.axvline(x=ridge_point, color='gray', linestyle=':', alpha=0.7)
    ax.text(ridge_point * 1.1, 5, f'Ridge Point\n({ridge_point:.0f} FLOPs/B)', 
            fontsize=8, color='gray')
    
    # Operations data
    operations = {
        'QKV Projection': {'ai': 379.21, 'achieved': 591.4 / 1000},  # GFLOPS -> TFLOPS
        'Attention Scores': {'ai': 85.33, 'achieved': 170 / 1000},   # Estimated
        'Attention Output': {'ai': 87.00, 'achieved': 175 / 1000},   # Estimated
        'Output Projection': {'ai': 372.36, 'achieved': 310 / 1000}, # Estimated
    }
    
    # Softer, professional colors
    colors = ['#c75a5a', '#5a8ac7', '#5ac77a', '#8a5ac7']
    markers = ['o', 's', '^', 'D']
    
    for (name, data), color, marker in zip(operations.items(), colors, markers):
        ax.scatter(data['ai'], data['achieved'], s=120, c=color, marker=marker, 
                   label=name, zorder=5, edgecolors='#333333', linewidths=0.8)
    
    # Annotations for bound regions with softer colors
    ax.fill_between([1, ridge_point], [0.1, 0.1], [1000, 1000], 
                    alpha=0.08, color='#d4a55a', label='Memory Bound')
    ax.fill_between([ridge_point, 1000], [0.1, 0.1], [1000, 1000], 
                    alpha=0.08, color='#5ad4a5', label='Compute Bound')
    
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylabel('Performance (TFLOPS)')
    ax.set_xlim(10, 1000)
    ax.set_ylim(1, 500)
    ax.set_title('Roofline Analysis: Attention Layer Operations on A100')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_roofline.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_roofline.png'))
    print("Saved: fig_roofline.pdf, fig_roofline.png")
    plt.close()


def fig2_kernel_comparison():
    """
    Bar chart comparing kernel implementations.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data from benchmarks
    implementations = ['PyTorch\nBaseline', 'Phase 2\n(Custom CUDA)', 'Phase 3\n(Hybrid)']
    latencies = [0.073, 0.110, 0.097]  # ms
    speedups = [1.0, 0.66, 0.75]  # relative to baseline
    
    # Softer, professional colors
    colors = ['#7fb685', '#d4a5a5', '#d4c9a5']  # soft green, soft red, soft yellow
    
    x = np.arange(len(implementations))
    bars = ax.bar(x, latencies, color=colors, edgecolor='#555555', linewidth=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, lat, spd in zip(bars, latencies, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.004,
                f'{lat:.3f} ms\n({spd:.2f}×)',
                ha='center', va='bottom', fontsize=9)
    
    # Reference line for baseline
    ax.axhline(y=0.073, color='#7fb685', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('QKV Projection Kernel Performance\n(B=4, S=512, Qwen3-30B Config)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(implementations)
    ax.set_ylim(0, 0.145)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation
    ax.annotate('↓ Lower is better', xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, style='italic', color='#666666')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_kernel_comparison.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_kernel_comparison.png'))
    print("Saved: fig_kernel_comparison.pdf, fig_kernel_comparison.png")
    plt.close()


def fig3_latency_breakdown():
    """
    Pie chart of layer-level latency breakdown.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data from benchmark analysis (B=1, S=512, decode)
    components = ['MoE FFN', 'Attention', 'QKV Proj.', 'Output Proj.', 'Other']
    sizes = [79, 10, 5.3, 2.8, 2.9]
    
    # Softer, more professional color palette
    colors = ['#d4a5a5', '#a5c4d4', '#a5d4b3', '#c4a5d4', '#d4d4d4']
    explode = (0.03, 0, 0, 0, 0)  # Slight explode for MoE slice
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 4 else '',
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.75
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # Add legend instead of labels on pie (avoids overlap)
    ax.legend(wedges, [f'{comp} ({size}%)' for comp, size in zip(components, sizes)],
              title="Components",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=9)
    
    ax.set_title('Decode Latency Attribution\n(Qwen3-30B-A3B, B=1, S=512)', fontsize=11, fontweight='bold')
    
    # Add annotation box
    textstr = 'QKV Projection: 5.3%\n→ Max gain from\n    optimization: ~4.6%'
    props = dict(boxstyle='round,pad=0.4', facecolor='#f5f5dc', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_latency_breakdown.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_latency_breakdown.png'))
    print("Saved: fig_latency_breakdown.pdf, fig_latency_breakdown.png")
    plt.close()


def fig4_component_breakdown():
    """
    Stacked bar chart showing component breakdown of kernel implementations.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Data from profiling
    components = ['GEMM', 'Bias Addition', 'Split', 'Transpose']
    
    # Times in ms for each implementation
    pytorch_times = [0.063, 0.002, 0.004, 0.004]  # F.linear fuses bias
    phase2_times = [0.063, 0.030, 0.009, 0.008]   # Custom kernel overhead
    phase3_times = [0.065, 0.030, 0.003, 0.003]   # PyTorch split/transpose
    
    x = np.arange(3)
    width = 0.55
    
    # Softer, professional colors
    colors = ['#a5c4d4', '#d4a5a5', '#a5d4b3', '#c4a5d4']
    
    implementations = ['PyTorch\nBaseline', 'Phase 2\n(Custom)', 'Phase 3\n(Hybrid)']
    all_times = [pytorch_times, phase2_times, phase3_times]
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        times = [all_times[j][i] for j in range(3)]
        bars = ax.bar(x, times, width, bottom=[sum(all_times[j][:i]) for j in range(3)],
                     label=comp, color=color, edgecolor='white', linewidth=1)
    
    # Add total labels
    totals = [sum(t) for t in all_times]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.004, f'{total:.3f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Component-Level Latency Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(implementations)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 0.145)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight bias difference with softer color
    ax.annotate('', xy=(0, 0.065), xytext=(1, 0.093),
                arrowprops=dict(arrowstyle='->', color='#b56b6b', lw=1.5))
    ax.text(0.5, 0.082, 'Bias overhead:\n+28 µs', ha='center', fontsize=8, color='#8b4444')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_component_breakdown.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_component_breakdown.png'))
    print("Saved: fig_component_breakdown.pdf, fig_component_breakdown.png")
    plt.close()


def main():
    print("Generating figures for QKV Fusion Final Report...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    fig1_roofline()
    fig2_kernel_comparison()
    fig3_latency_breakdown()
    fig4_component_breakdown()
    
    print("\nAll figures generated successfully!")
    print("\nFigure descriptions for report:")
    print("  - fig_roofline: Roofline diagram with attention operations plotted")
    print("  - fig_kernel_comparison: Bar chart of implementation latencies")
    print("  - fig_latency_breakdown: Pie chart showing MoE dominance (79%)")
    print("  - fig_component_breakdown: Stacked bars showing bias overhead")


if __name__ == "__main__":
    main()

