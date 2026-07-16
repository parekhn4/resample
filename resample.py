"""
Usage:
    python resample.py                        
    python resample.py --era 2016preVFP
    python resample.py --era 2016preVFP --sample GGJets
    python resample.py --run-id 20260715_143022 --era 2017    
    python resample.py --config /path/to/config.yaml
"""
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def import_prediction(base_dir):
    """Import prediction.py's pure functions/classes (MLP, load_metadata,
    load_models, build_X, standardize, infer, get_fold_assignment) without
    relying on its own hardcoded SAMPLE_BASE/MODEL_BASE constants -- all
    paths here come from config.yaml instead."""
    sys.path.insert(0, base_dir)
    import prediction as pred
    return pred


def apply_presel(X_raw, feature_names, presel_cuts):
    """Boolean mask from a list of {column, op, value} cuts applied to raw
    (unstandardized) values. Events with raw sentinel values (e.g. -999)
    in a cut column naturally fail the cut and get excluded."""
    mask = np.ones(X_raw.shape[0], dtype=bool)
    for cut in presel_cuts:
        idx = feature_names.index(cut["column"])
        vals = X_raw[:, idx]
        op, value = cut["op"], cut["value"]
        if op == ">":
            mask &= (vals > value)
        elif op == "<":
            mask &= (vals < value)
        elif op == ">=":
            mask &= (vals >= value)
        elif op == "<=":
            mask &= (vals <= value)
        else:
            raise ValueError(f"Unsupported op '{op}' in presel_cuts")
    return mask


def build_hist(values, fill_nan, bins):
    """Sentinel-filtered histogram of a standardized column, used as the
    resampling source distribution."""
    vals = values[values != fill_nan]
    counts, edges = np.histogram(vals, bins=bins)
    total = counts.sum()
    if total == 0:
        raise RuntimeError("Signal histogram has zero entries after sentinel filtering.")
    probs = counts / total
    return edges, probs


def sample_from_hist(edges, probs, n, rng):
    """Draw n values from a histogram: pick a bin weighted by its
    probability, then a uniform position within that bin."""
    bin_idx = rng.choice(len(probs), size=n, p=probs)
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    return lo + rng.random(n) * (hi - lo)


def list_eras(sample_base):
    return sorted(
        d for d in os.listdir(sample_base)
        if os.path.isdir(os.path.join(sample_base, d))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resample backgrounds from HH signal shape and score with the two-fold MLP.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--era", default=None, help="Restrict to a single era")
    parser.add_argument("--sample", default=None, help="Restrict to a single background sample")
    parser.add_argument("--ext", default=None, help="Override the ext tag used in output filenames")
    parser.add_argument("--run-id", default=None,
                         help="Timestamped subdirectory under output_base to write into "
                              "(default: create a new one now, e.g. 20260715_143022). "
                              "Pass the same --run-id across multiple invocations "
                              "(different --era/--sample) to accumulate into one run.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_dir = cfg["paths"]["base_dir"]
    sample_base = os.path.join(base_dir, cfg["paths"]["sample_base"])
    model_base = os.path.join(base_dir, cfg["paths"]["model_base"])
    output_base = cfg["paths"]["output_base"]

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    ext = args.ext or cfg["model"]["ext"]
    combo_name = cfg["resample"]["combo_name"]
    fill_nan = cfg["model"]["fill_nan"]
    n_classes = cfg["model"]["n_classes"]
    batch_size = cfg["model"]["batch_size"]
    class_names = cfg["class_names"]
    seed = cfg["resample"]["seed"]

    hparams = {
        "num_layers": cfg["model"]["num_layers"],
        "num_nodes": cfg["model"]["num_nodes"],
        "act_fn_name": cfg["model"]["act_fn"],
        "dropout_prob": cfg["model"]["dropout_prob"],
    }

    pred = import_prediction(base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_names, mean, std, k_folds = pred.load_metadata(model_base)
    print(f"Loaded {len(feature_names)} features, k_folds={k_folds}")

    print("Loading models ...")
    models = pred.load_models(model_base, k_folds, hparams, n_classes, device)

    signal_sample = cfg["resample"]["signal_sample"]
    background_samples = [args.sample] if args.sample else cfg["resample"]["background_samples"]

    hist_var_cfgs = cfg["resample"]["hist_vars"]          # list of {name, bins}
    fixed_value = cfg["resample"]["fixed_vars"]["value"]
    fixed_columns = cfg["resample"]["fixed_vars"]["columns"]

    output_vars = cfg["output_vars"]                      # list of {column, out_name}
    presel_cuts = cfg["resample"]["presel_cuts"]

    eras = [args.era] if args.era else list_eras(sample_base)

    for era in eras:
        signal_pq = os.path.join(sample_base, era, signal_sample, "events.parquet")
        if not os.path.exists(signal_pq):
            print(f"[{era}] signal file missing, skipping era: {signal_pq}")
            continue

        print(f"[{era}] building signal histograms from {signal_sample} ...")
        signal_X = pred.build_X(signal_pq, feature_names)
        signal_presel_mask = apply_presel(signal_X, feature_names, presel_cuts)
        print(f"    presel: {signal_presel_mask.sum()}/{len(signal_presel_mask)} signal events pass")
        signal_X = signal_X[signal_presel_mask]
        signal_X_std = pred.standardize(signal_X, mean, std, fill_nan)

        hists = {}
        for hv in hist_var_cfgs:
            col_idx = feature_names.index(hv["name"])
            edges, probs = build_hist(signal_X_std[:, col_idx], fill_nan, hv["bins"])
            hists[hv["name"]] = (col_idx, edges, probs)
            print(f"    {hv['name']}: {len(probs)} bins, range [{edges[0]:.3f}, {edges[-1]:.3f}]")

        fixed_col_idxs = [feature_names.index(c) for c in fixed_columns]

        for sample in background_samples:
            sample_dir = os.path.join(sample_base, era, sample)
            bkg_pq = os.path.join(sample_dir, "events.parquet")
            if not os.path.exists(bkg_pq):
                print(f"[{era}/{sample}] events.parquet missing, skipping")
                continue

            print(f"[{era}/{sample}] resampling + scoring ...")
            rng = np.random.default_rng(seed)

            X_bkg = pred.build_X(bkg_pq, feature_names)
            bkg_presel_mask = apply_presel(X_bkg, feature_names, presel_cuts)
            print(f"    presel: {bkg_presel_mask.sum()}/{len(bkg_presel_mask)} background events pass")
            X_bkg = X_bkg[bkg_presel_mask]
            X_bkg_std = pred.standardize(X_bkg, mean, std, fill_nan)
            n = X_bkg_std.shape[0]

            for var_name, (col_idx, edges, probs) in hists.items():
                X_bkg_std[:, col_idx] = sample_from_hist(edges, probs, n, rng)

            for col_idx in fixed_col_idxs:
                X_bkg_std[:, col_idx] = fixed_value

            fold_assignment = pred.get_fold_assignment(bkg_pq, k_folds)[bkg_presel_mask]
            y_full = np.empty((n, n_classes), dtype=np.float32)
            for fi, model in models.items():
                target = (fi - 1 + k_folds) % k_folds
                mask = (fold_assignment == target)
                if mask.sum() == 0:
                    continue
                y_full[mask] = pred.infer(model, X_bkg_std[mask], device, batch_size)

            raw_cols = ["event"] + [ov["column"] for ov in output_vars]
            raw_table = pq.read_table(bkg_pq, columns=raw_cols)
            out = {"event": raw_table.column("event").to_numpy(zero_copy_only=False)[bkg_presel_mask]}
            for ov in output_vars:
                out[ov["out_name"]] = raw_table.column(ov["column"]).to_numpy(zero_copy_only=False)[bkg_presel_mask]
            for i, cname in enumerate(class_names):
                out[cname] = y_full[:, i]

            out_df = pd.DataFrame(out)

            out_dir = os.path.join(run_dir, "individual_samples", era, sample)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"y_resampled_{combo_name}_{ext}.parquet")
            out_df.to_parquet(out_path, index=False)
            print(f"[{era}/{sample}] wrote {out_path}  ({n} events)")

    print("Done.")


if __name__ == "__main__":
    main()