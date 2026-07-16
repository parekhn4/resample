"""
Usage:
    python sculpting_check.py                      
    python sculpting_check.py --run-id 20260715_143022
    python sculpting_check.py --era 2016preVFP           
    python sculpting_check.py --config /path/to/config.yaml
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    HAVE_MPLHEP = True
except ImportError:
    HAVE_MPLHEP = False

CMAP = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
        "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_run_dir(output_base, run_id):
    """Return output_base/run_id if given, else the most recently created
    run directory under output_base (timestamp-named dirs sort correctly
    as strings, e.g. 20260715_143022)."""
    if run_id:
        run_dir = os.path.join(output_base, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    candidates = sorted(
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and d != "plots"
    )
    if not candidates:
        raise RuntimeError(f"No run directories found under {output_base}. Run resample.py first.")
    latest = candidates[-1]
    print(f"No --run-id given, using most recent run: {latest}")
    return os.path.join(output_base, latest)


def apply_cut(df, cut_dict):
    mask = np.ones(len(df), dtype=bool)
    for col, spec in cut_dict.items():
        op, value = spec["op"], spec["value"]
        if op == ">":
            mask &= (df[col].to_numpy() > value)
        elif op == "<":
            mask &= (df[col].to_numpy() < value)
        else:
            raise ValueError(f"Unsupported op '{op}' for column '{col}'")
    return mask


def load_all(run_dir, combo_name, ext, eras=None, samples=None):
    pattern = os.path.join(run_dir, "individual_samples", "*", "*",
                            f"y_resampled_{combo_name}_{ext}.parquet")
    paths = sorted(glob.glob(pattern))

    dfs = []
    for p in paths:
        parts = p.split(os.sep)
        sample, era = parts[-2], parts[-3]
        if eras and era not in eras:
            continue
        if samples and sample not in samples:
            continue
        df = pd.read_parquet(p)
        df["era"] = era
        df["sample"] = sample
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No resampled parquets found matching {pattern}")
    return pd.concat(dfs, ignore_index=True)


def plot_panel(ax, df, cuts, plot_var):
    bin_edges = np.linspace(plot_var["range"][0], plot_var["range"][1], plot_var["bins"] + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
 
    for (cut_name, cut_dict), color in zip(cuts.items(), CMAP):
        mask = apply_cut(df, cut_dict)
        vals = df.loc[mask, plot_var["name"]].to_numpy()
        n_pass = len(vals)
        counts, _ = np.histogram(vals, bins=bin_edges)
        total = counts.sum()
        if total == 0:
            print(f"    [{plot_var['name']}] cut '{cut_name}': 0 events passing, skipping curve")
            continue
        density = counts / (total * bin_width)
        err = np.sqrt(counts) / (total * bin_width)
        ax.stairs(density, bin_edges, label=f"{cut_name} (n={n_pass})", color=color, linewidth=1.8)
        ax.errorbar(bin_centers, density, yerr=err, fmt="none", color=color, alpha=0.6)
 
    if HAVE_MPLHEP:
        hep.cms.text("Preliminary", ax=ax)
    ax.set_xlabel(f"{plot_var['name']} [GeV] / {bin_width:.1f} GeV")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc="upper right")
 
 
def make_plot(df, cuts, plot_vars, out_path, show=False):
    fig, axes = plt.subplots(1, len(plot_vars), figsize=(8 * len(plot_vars), 7))
    if len(plot_vars) == 1:
        axes = [axes]
 
    for ax, plot_var in zip(axes, plot_vars):
        plot_panel(ax, df, cuts, plot_var)
 
    fig.suptitle("Sculpting check", fontsize=16, y=1.02)
    fig.tight_layout()
 
    if show:
        plt.show()
    else:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Apply score cuts to resampled background and plot mass sculpting.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--era", default=None, help="Restrict to a single era")
    parser.add_argument("--sample", default=None, help="Restrict to a single background sample")
    parser.add_argument("--ext", default=None)
    parser.add_argument("--run-id", default=None,
                         help="Timestamped run directory under output_base to read from "
                              "(default: most recently created one)")
    parser.add_argument("--outdir", default=None, help="Where to save plots (default: <run_dir>/plots)")
    parser.add_argument("--show", action="store_true", help="Display plots instead of saving")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_base = cfg["paths"]["output_base"]
    combo_name = cfg["resample"]["combo_name"]
    ext = args.ext or cfg["model"]["ext"]

    run_dir = resolve_run_dir(output_base, args.run_id)
    print(f"Run directory: {run_dir}")

    eras = [args.era] if args.era else None
    samples = [args.sample] if args.sample else None

    print("Loading resampled+scored parquets ...")
    df = load_all(run_dir, combo_name, ext, eras=eras, samples=samples)
    print(f"Loaded {len(df)} events across {df['sample'].nunique()} samples, {df['era'].nunique()} eras")

    outdir = args.outdir or os.path.join(run_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    cuts = cfg["cuts"]
    plot_vars = cfg["plot_vars"]
    print(f"Plotting {[pv['name'] for pv in plot_vars]} ...")
    out_path = os.path.join(outdir, f"sculpting_{combo_name}_{ext}.pdf")
    make_plot(df, cuts, plot_vars, out_path, show=args.show)

    print("Done.")


if __name__ == "__main__":
    main()