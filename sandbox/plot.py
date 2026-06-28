"""
plot_results.py
---------------
Reads an H2 DFT convergence results file and produces two plots:
  1. |E_ibs - E_exact| and |E_cd - E_exact| vs N  (log-scale y-axis)
  2. Execution times (ibs and cd) vs N
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_file(path: str):
    """Return (E_exact, N, E_ibs, E_cd, t_ibs, t_cd) from the results file."""
    with open(path) as fh:
        text = fh.read()

    # extract the exact KS energy
    m = re.search(r"correct KS energy is\s*=\s*([-\d.]+)", text)
    if not m:
        raise ValueError("Could not find 'correct KS energy' in the file.")
    E_exact = float(m.group(1))

    # parse the data table
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].lstrip("-").isdigit():
            try:
                n, e_ibs, e_cd, t_ibs, t_cd = (float(p) for p in parts)
                rows.append((int(n), e_ibs, e_cd, t_ibs, t_cd))
            except ValueError:
                pass

    if not rows:
        raise ValueError("No data rows found in the file.")

    rows.sort(key=lambda r: r[0])
    N     = np.array([r[0] for r in rows])
    E_ibs = np.array([r[1] for r in rows])
    E_cd  = np.array([r[2] for r in rows])
    t_ibs = np.array([r[3] for r in rows])
    t_cd  = np.array([r[4] for r in rows])
    return E_exact, N, E_ibs, E_cd, t_ibs, t_cd


def make_plots(path: str, out_prefix: str = "results"):
    E_exact, N, E_ibs, E_cd, t_ibs, t_cd = parse_file(path)

    err_ibs = np.abs(E_ibs - E_exact)
    err_cd  = np.abs(E_cd  - E_exact)

    # ── colour palette ────────────────────────────────────────────────────────
    CLR_IBS  = "#2563EB"   # blue  – ibs
    CLR_CD   = "#DC2626"   # red   – cd
    CLR_BG   = "#F8FAFC"
    CLR_GRID = "#CBD5E1"
    CLR_TEXT = "#1E293B"

    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "axes.facecolor"   : CLR_BG,
        "figure.facecolor" : "white",
        "axes.edgecolor"   : "#94A3B8",
        "axes.labelcolor"  : CLR_TEXT,
        "xtick.color"      : CLR_TEXT,
        "ytick.color"      : CLR_TEXT,
        "text.color"       : CLR_TEXT,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : True,
        "grid.color"       : CLR_GRID,
        "grid.linestyle"   : "--",
        "grid.linewidth"   : 0.7,
        "grid.alpha"       : 0.8,
        "legend.framealpha": 0.9,
        "legend.edgecolor" : CLR_GRID,
    })

    title = f"H₂  def2-TZVPPD   |  $E_{{\\rm exact}}$ = {E_exact:.8f} Eh"

    # ── Plot 1 : energy errors (log scale) ───────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.semilogy(N, err_ibs, "o-", color=CLR_IBS, lw=2, ms=6,
                 label=r"$|E_{\rm ibs} - E_{\rm exact}|$")
    ax1.semilogy(N, err_cd,  "s-", color=CLR_CD,  lw=2, ms=6,
                 label=r"$|E_{\rm cd} - E_{\rm exact}|$")

    ax1.set_xlabel("Iteration  $N$", fontsize=11)
    ax1.set_ylabel("Absolute energy error  (Eh)", fontsize=11)
    ax1.set_title("Energy convergence", fontsize=12, fontweight="bold")
    ax1.set_xlim(left=0.5)
    all_err = np.concatenate([err_ibs, err_cd])
    ax1.set_ylim(all_err.min() * 0.5, all_err.max() * 2)
    ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    fig1.tight_layout()
    out1 = f"{out_prefix}_energy.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved → {out1}")

    # ── Plot 2 : timings ─────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.plot(N, t_ibs, "o-", color=CLR_IBS, lw=2, ms=6, label="ibs  time")
    ax2.plot(N, t_cd,  "s-", color=CLR_CD,  lw=2, ms=6, label="cd   time")

    ax2.set_xlabel("Iteration  $N$", fontsize=11)
    ax2.set_ylabel("Wall time  (s)", fontsize=11)
    ax2.set_title("Execution times", fontsize=12, fontweight="bold")
    ax2.set_xlim(left=0.5)
    all_t = np.concatenate([t_ibs, t_cd])
    ax2.set_ylim(all_t.min() * 0.9, all_t.max() * 1.05)
    ax2.legend(fontsize=10)

    fig2.tight_layout()
    out2 = f"{out_prefix}_timings.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path = Path(__file__).parent / "formaldehyde" / "formaldehyde_def2TZVP_results.txt"
    make_plots(input_path)

