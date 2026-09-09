#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def get_combo_names(cfg):
    resample_cfg = cfg["resample"]
    if "combos" in resample_cfg:
        return [c["name"] for c in resample_cfg["combos"]]
    return [resample_cfg["combo_name"]]


def resolve_run_dir(output_base, run_id):
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


def resolve_cuts(df, cfg):
    cuts_cfg = cfg["cuts"]

    if isinstance(cuts_cfg, dict):
        names = list(cuts_cfg.keys())
        masks = {name: apply_cut(df, cuts_cfg[name]) for name in names}
        return names, masks

    shared_expr = cfg.get("shared_cut")
    shared_mask = df.eval(shared_expr).to_numpy() if shared_expr else np.ones(len(df), dtype=bool)

    names = []
    masks = {}
    assigned = np.zeros(len(df), dtype=bool)
    for entry in cuts_cfg:
        name = entry["name"]
        expr = entry.get("expr")
        names.append(name)
        if expr is None:
            masks[name] = np.ones(len(df), dtype=bool)
            continue
        raw_mask = df.eval(expr).to_numpy() & shared_mask
        this_mask = raw_mask & ~assigned
        masks[name] = this_mask
        assigned |= this_mask
    return names, masks


def load_all(run_dir, combo_name, ext, eras=None, samples=None, exclude_samples=None):
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
        if exclude_samples and sample in exclude_samples:
            continue
        df = pd.read_parquet(p)
        df["era"] = era
        df["sample"] = sample
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No resampled parquets found matching {pattern}")
    return pd.concat(dfs, ignore_index=True)


def plot_panel(ax, df, cfg, plot_var, weighted):
    bin_edges = np.linspace(plot_var["range"][0], plot_var["range"][1], plot_var["bins"] + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    cut_names, cut_masks = resolve_cuts(df, cfg)

    for cut_name, color in zip(cut_names, CMAP):
        mask = cut_masks[cut_name]
        vals = df.loc[mask, plot_var["name"]].to_numpy()
        n_pass = len(vals)
        if n_pass == 0:
            print(f"    [{plot_var['name']}] cut '{cut_name}': 0 events passing, skipping curve")
            continue
        if weighted:
            w = df.loc[mask, "weight_tot"].to_numpy()
            counts, _ = np.histogram(vals, bins=bin_edges, weights=w)
            w2, _ = np.histogram(vals, bins=bin_edges, weights=w ** 2)
            total = counts.sum()
            if total == 0:
                print(f"    [{plot_var['name']}] cut '{cut_name}': sum(weight)==0, skipping curve")
                continue
            if total < 0:
                print(f"    [{plot_var['name']}] cut '{cut_name}': sum(weight)={total:.3g} < 0 "
                      f"(negative-weight events dominate this range), skipping curve")
                continue
            density = counts / (total * bin_width)
            err = np.sqrt(w2) / (total * bin_width)
            label = f"{cut_name} (n={n_pass}, sum(w)={w.sum():.3g})"
        else:
            counts, _ = np.histogram(vals, bins=bin_edges)
            total = counts.sum()
            density = counts / (total * bin_width)
            err = np.sqrt(counts) / (total * bin_width)
            label = f"{cut_name} (n={n_pass})"
        ax.stairs(density, bin_edges, label=label, color=color, linewidth=1.8)
        ax.errorbar(bin_centers, density, yerr=err, fmt="none", color=color, alpha=0.6)

    if HAVE_MPLHEP:
        hep.cms.text("Preliminary", ax=ax)
    ax.set_xlabel(f"{plot_var['name']} [GeV] / {bin_width:.1f} GeV")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc="upper right")


def build_figure(df, cfg, plot_vars, page_title, weighted, cuts_source):
    fig, axes = plt.subplots(1, len(plot_vars), figsize=(8 * len(plot_vars), 7))
    if len(plot_vars) == 1:
        axes = [axes]

    for ax, plot_var in zip(axes, plot_vars):
        plot_panel(ax, df, cfg, plot_var, weighted)

    samples_str = ", ".join(sorted(df["sample"].unique()))
    info = (
        f"model: {cfg['model']['ext']}\n"
        f"samples: {samples_str}\n"
        f"weighted: {weighted}\n"
        f"cuts: {cuts_source}"
    )
    fig.text(0.01, 1.06, info, fontsize=8, family="monospace", va="top", ha="left",
              transform=fig.transFigure,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    fig.suptitle(page_title, fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Apply score cuts to resampled background and plot mass sculpting, "
                    "one page per resample combo.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--era", default=None, help="Restrict to a single era")
    parser.add_argument("--sample", default=None, help="Restrict to a single background sample "
                         "(legacy alias -- prefer --include-sample, which is repeatable)")
    parser.add_argument("--include-sample", action="append", default=None,
                         help="Only include this sample by name (repeatable, e.g. "
                              "--include-sample GGJets --include-sample DDQCDGJET). Opt-in "
                              "whitelist, the mirror of --exclude-sample's opt-out blacklist -- "
                              "if omitted, all samples are included except any given via "
                              "--exclude-sample.")
    parser.add_argument("--exclude-sample", action="append", default=None,
                         help="Exclude a sample by name from the plots without touching its "
                              "resampled parquets on disk (repeatable, e.g. --exclude-sample TTGG)")
    parser.add_argument("--ext", default=None)
    parser.add_argument("--run-id", default=None,
                         help="Timestamped run directory under output_base to read from "
                              "(default: most recently created one)")
    parser.add_argument("--outdir", default=None, help="Where to save plots (default: <run_dir>/plots)")
    parser.add_argument("--cuts-file", default=None,
                         help="YAML file with a cuts: block (and optionally shared_cut:) that "
                              "overrides --config's own, without duplicating the whole config -- "
                              "e.g. for per-score scans against the same already-scored run.")
    parser.add_argument("--suffix", default=None,
                         help="Bookkeeping tag appended to the output PDF filename as _<suffix> "
                              "(e.g. --suffix no_ttgg -> sculpting_<ext>_no_ttgg.pdf), so multiple "
                              "labeled variants can coexist in the same run's plots/ dir instead of "
                              "overwriting each other.")
    parser.add_argument("--show", action="store_true", help="Display plots instead of saving")
    parser.add_argument("--weight-mode", choices=["both", "weighted", "unweighted"], default="both",
                         help="Which histogram variant(s) to produce as separate PDFs: raw event "
                              "counts (unweighted), weight_tot-weighted, or both (default: both).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cuts_file:
        cuts_override = load_config(args.cuts_file)
        cfg["cuts"] = cuts_override["cuts"]
        if "shared_cut" in cuts_override:
            cfg["shared_cut"] = cuts_override["shared_cut"]
        print(f"Cuts overridden from {args.cuts_file}")
    output_base = cfg["paths"]["output_base"]
    ext = args.ext or cfg["model"]["ext"]

    run_dir = resolve_run_dir(output_base, args.run_id)
    print(f"Run directory: {run_dir}")

    eras = [args.era] if args.era else None
    include_samples = list(args.include_sample) if args.include_sample else []
    if args.sample:
        include_samples.append(args.sample)
    samples = set(include_samples) if include_samples else None
    exclude_samples = set(args.exclude_sample) if args.exclude_sample else None
    plot_vars = cfg["plot_vars"]

    combo_names = get_combo_names(cfg)
    print(f"Combos ({len(combo_names)} page(s)): {combo_names}")

    outdir = args.outdir or os.path.join(run_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    suffix_tag = f"_{args.suffix}" if args.suffix else ""
    cuts_source = os.path.basename(args.cuts_file) if args.cuts_file else os.path.basename(args.config)

    combo_dfs = {}
    for combo_name in combo_names:
        try:
            combo_dfs[combo_name] = load_all(run_dir, combo_name, ext, eras=eras, samples=samples, exclude_samples=exclude_samples)
        except RuntimeError as e:
            print(f"  [combo '{combo_name}'] SKIPPED: {e}")

    has_weight = any("weight_tot" in df.columns for df in combo_dfs.values())

    modes = {"both": ["unweighted", "weighted"], "weighted": ["weighted"], "unweighted": ["unweighted"]}[args.weight_mode]
    if "weighted" in modes and not has_weight:
        print("No weight_tot column found in any loaded combo -- skipping weighted variant.")
        modes = [m for m in modes if m != "weighted"]
    if not modes:
        raise RuntimeError("No plot variant to produce (weighted requested but weight_tot missing).")

    for mode in modes:
        weighted = (mode == "weighted")
        out_path = os.path.join(outdir, f"sculpting_{ext}_{mode}{suffix_tag}.pdf")
        pdf = None if args.show else PdfPages(out_path)
        try:
            for combo_name in combo_names:
                if combo_name not in combo_dfs:
                    continue
                df = combo_dfs[combo_name]
                print(f"  [{mode}] combo '{combo_name}': {len(df)} events across "
                      f"{df['sample'].nunique()} samples, {df['era'].nunique()} eras")
                fig = build_figure(df, cfg, plot_vars, page_title=f"Sculpting check ({mode}) -- {combo_name}", weighted=weighted, cuts_source=cuts_source)
                if args.show:
                    plt.show()
                    plt.close(fig)
                else:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        finally:
            if pdf is not None:
                pdf.close()

        if not args.show:
            print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
