#!/usr/bin/env python3
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
    sys.path.insert(0, base_dir)
    import prediction as pred
    return pred


def get_fold_values_legacy(event_raw, k_folds):
    return (event_raw.astype(np.int64) % k_folds).astype(np.int32)


def get_fold_values_bhive_new(event_raw, k_folds):
    return ((event_raw.astype(np.int64) // 2) % k_folds).astype(np.int32)


def get_fold_values_detsplit_heldout(event_raw, split_k, held_out_folds):
    # events outside held_out_folds get -1 and are dropped, never scored
    fold_assignment = ((event_raw.astype(np.int64) // 2) % split_k).astype(np.int32)
    is_held_out = np.isin(fold_assignment, held_out_folds)
    return np.where(is_held_out, 0, -1).astype(np.int32)


def get_fold_assignment(pq_path, k_folds, fold_convention, split_k=None, held_out_folds=None):
    table = pq.read_table(pq_path, columns=["event"])
    event_raw = table.column("event").to_numpy(zero_copy_only=False)
    if fold_convention == "bhive_new":
        return get_fold_values_bhive_new(event_raw, k_folds)
    elif fold_convention == "legacy":
        return get_fold_values_legacy(event_raw, k_folds)
    elif fold_convention == "detsplit_heldout":
        return get_fold_values_detsplit_heldout(event_raw, split_k, held_out_folds)
    else:
        raise ValueError(f"Unknown fold_convention '{fold_convention}' (expected 'legacy', 'bhive_new', or 'detsplit_heldout')")


def fold_routing_target(fi, k_folds, fold_convention):
    if fold_convention == "bhive_new":
        return fi
    elif fold_convention == "legacy":
        return (fi - 1 + k_folds) % k_folds
    elif fold_convention == "detsplit_heldout":
        return 0
    else:
        raise ValueError(f"Unknown fold_convention '{fold_convention}' (expected 'legacy', 'bhive_new', or 'detsplit_heldout')")


def apply_presel(X_raw, feature_names, presel_cuts):
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


def resolve_combos(cfg, feature_names):
    resample_cfg = cfg["resample"]
    if "combos" in resample_cfg:
        raw_combos = resample_cfg["combos"]
    else:
        raw_combos = [{
            "name": resample_cfg["combo_name"],
            "hist_vars": resample_cfg["hist_vars"],
            "fixed_vars": resample_cfg["fixed_vars"]["columns"],
        }]

    resolved = []
    cum_hist = {}
    cum_fixed = []
    for stage in raw_combos:
        for hv in stage.get("hist_vars", []):
            name = hv["name"]
            if name not in feature_names:
                print(f"  [combo '{stage['name']}'] WARNING: '{name}' not a model input, skipping")
                continue
            cum_hist[name] = hv["bins"]
        for name in stage.get("fixed_vars", []):
            if name not in feature_names:
                print(f"  [combo '{stage['name']}'] WARNING: '{name}' not a model input, skipping")
                continue
            if name not in cum_fixed:
                cum_fixed.append(name)
        resolved.append({
            "name": stage["name"],
            "hist_vars": dict(cum_hist),
            "fixed_vars": list(cum_fixed),
        })
    return resolved


def build_hist(values, fill_nan, bins):
    vals = values[values != fill_nan]
    counts, edges = np.histogram(vals, bins=bins)
    total = counts.sum()
    if total == 0:
        raise RuntimeError("Signal histogram has zero entries after sentinel filtering")
    probs = counts / total
    return edges, probs


def sample_from_hist(edges, probs, n, rng):
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
        description="Resample backgrounds from HH signal shape and score with the model, "
                    "one output file per resample combo ('page').")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--era", default=None, help="Restrict to a single era")
    parser.add_argument("--sample", default=None, help="Restrict to a single background sample")
    parser.add_argument("--ext", default=None, help="Override the ext tag used in output filenames")
    parser.add_argument("--run-id", default=None,
                         help="Timestamped subdirectory under output_base to write into "
                              "(default: create a new one now). Reuse the same --run-id "
                              "across invocations to accumulate into one run.")
    parser.add_argument("--suffix", default=None,
                         help="Bookkeeping tag appended to the run-id as _<suffix>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resample_cfg = cfg["resample"]

    base_dir = cfg["paths"]["base_dir"]
    sample_base = os.path.join(base_dir, cfg["paths"]["sample_base"])
    model_base = os.path.join(base_dir, cfg["paths"]["model_base"])
    output_base = cfg["paths"]["output_base"]
    events_filename = cfg["paths"].get("events_filename", "events.parquet")

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.suffix:
        run_id = f"{run_id}_{args.suffix}"
    run_dir = os.path.join(output_base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    ext = args.ext or cfg["model"]["ext"]
    fill_nan = cfg["model"]["fill_nan"]
    n_classes = cfg["model"]["n_classes"]
    batch_size = cfg["model"]["batch_size"]
    class_names = cfg["class_names"]
    seed = resample_cfg["seed"]
    fold_convention = cfg["model"].get("fold_convention", "legacy")
    split_k = cfg["model"].get("split_k")
    held_out_folds = cfg["model"].get("held_out_folds")
    print(f"Fold convention: {fold_convention}"
          + (f" (split_k={split_k}, held_out_folds={held_out_folds})" if fold_convention == "detsplit_heldout" else ""))

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

    combos = resolve_combos(cfg, feature_names)
    print(f"Resample combos ({len(combos)} page(s)): {[c['name'] for c in combos]}")

    fixed_value = resample_cfg.get("fixed_value", resample_cfg.get("fixed_vars", {}).get("value", 1.0))

    all_hist_var_cfgs = {}
    all_fixed_names = []
    for c in combos:
        all_hist_var_cfgs.update(c["hist_vars"])
        for name in c["fixed_vars"]:
            if name not in all_fixed_names:
                all_fixed_names.append(name)
    fixed_col_idxs = {name: feature_names.index(name) for name in all_fixed_names}

    signal_sample = resample_cfg["signal_sample"]
    background_samples = [args.sample] if args.sample else resample_cfg["background_samples"]

    output_vars = cfg["output_vars"]
    presel_cuts = resample_cfg["presel_cuts"]

    eras = [args.era] if args.era else list_eras(sample_base)

    for era in eras:
        signal_pq = os.path.join(sample_base, era, signal_sample, events_filename)
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
        for name, bins in all_hist_var_cfgs.items():
            col_idx = feature_names.index(name)
            edges, probs = build_hist(signal_X_std[:, col_idx], fill_nan, bins)
            hists[name] = (col_idx, edges, probs)
            print(f"    {name}: {len(probs)} bins, range [{edges[0]:.3f}, {edges[-1]:.3f}]")

        for sample in background_samples:
            sample_dir = os.path.join(sample_base, era, sample)
            bkg_pq = os.path.join(sample_dir, events_filename)
            if not os.path.exists(bkg_pq):
                print(f"[{era}/{sample}] {events_filename} missing, skipping")
                continue

            print(f"[{era}/{sample}] resampling + scoring ...")

            X_bkg = pred.build_X(bkg_pq, feature_names)
            bkg_presel_mask = apply_presel(X_bkg, feature_names, presel_cuts)
            print(f"    presel: {bkg_presel_mask.sum()}/{len(bkg_presel_mask)} background events pass")
            X_bkg = X_bkg[bkg_presel_mask]
            X_bkg_std = pred.standardize(X_bkg, mean, std, fill_nan)
            n = X_bkg_std.shape[0]

            fold_assignment = get_fold_assignment(bkg_pq, k_folds, fold_convention,
                                                   split_k=split_k, held_out_folds=held_out_folds)[bkg_presel_mask]

            scored_mask = np.zeros(n, dtype=bool)
            for fi in models:
                target = fold_routing_target(fi, k_folds, fold_convention)
                scored_mask |= (fold_assignment == target)
            if not scored_mask.all():
                print(f"    fold filter ({fold_convention}): {scored_mask.sum()}/{n} events actually "
                      f"scored, {n - scored_mask.sum()} excluded (not covered by any fold target)")
                X_bkg_std = X_bkg_std[scored_mask]
                fold_assignment = fold_assignment[scored_mask]
                bkg_presel_mask_scored_idx = np.flatnonzero(bkg_presel_mask)[scored_mask]
                bkg_presel_mask = np.zeros_like(bkg_presel_mask)
                bkg_presel_mask[bkg_presel_mask_scored_idx] = True
                n = X_bkg_std.shape[0]

            has_weight = "weight_tot" in pq.read_schema(bkg_pq).names
            raw_cols = ["event"] + [ov["column"] for ov in output_vars] + (["weight_tot"] if has_weight else [])
            if not has_weight:
                print(f"[{era}/{sample}] WARNING: no weight_tot column, output won't carry weights")
            raw_table = pq.read_table(bkg_pq, columns=raw_cols)
            base_out = {"event": raw_table.column("event").to_numpy(zero_copy_only=False)[bkg_presel_mask]}
            for ov in output_vars:
                base_out[ov["out_name"]] = raw_table.column(ov["column"]).to_numpy(zero_copy_only=False)[bkg_presel_mask]
            if has_weight:
                base_out["weight_tot"] = raw_table.column("weight_tot").to_numpy(zero_copy_only=False)[bkg_presel_mask]

            rng = np.random.default_rng(seed)
            out_dir = os.path.join(run_dir, "individual_samples", era, sample)
            os.makedirs(out_dir, exist_ok=True)

            for combo in combos:
                X = X_bkg_std.copy()
                for name in combo["hist_vars"]:
                    col_idx, edges, probs = hists[name]
                    X[:, col_idx] = sample_from_hist(edges, probs, n, rng)
                for name in combo["fixed_vars"]:
                    X[:, fixed_col_idxs[name]] = fixed_value

                y_full = np.empty((n, n_classes), dtype=np.float32)
                for fi, model in models.items():
                    target = fold_routing_target(fi, k_folds, fold_convention)
                    mask = (fold_assignment == target)
                    if mask.sum() == 0:
                        continue
                    y_full[mask] = pred.infer(model, X[mask], device, batch_size)

                out = dict(base_out)
                for i, cname in enumerate(class_names):
                    out[cname] = y_full[:, i]

                out_df = pd.DataFrame(out)
                out_path = os.path.join(out_dir, f"y_resampled_{combo['name']}_{ext}.parquet")
                out_df.to_parquet(out_path, index=False)
                print(f"[{era}/{sample}] combo '{combo['name']}': wrote {out_path} "
                      f"({n} events, {len(combo['hist_vars'])} resampled + {len(combo['fixed_vars'])} fixed vars)")

    print("Done.")


if __name__ == "__main__":
    main()
