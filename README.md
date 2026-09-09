# resample

resample.py does the resampling and scoring. sculpting_check.py applies the cuts and makes the plots.

needs a prediction.py sitting at the model's own artifact directory, with load_metadata, load_models, build_X, standardize, infer. resample.py imports it via base_dir in the config.

## running it

```
python3 resample.py --config config_SnTDetSplit1D.yaml
python3 sculpting_check.py --config config_SnTDetSplit1D.yaml
```

first command scores everything and writes one parquet per era/sample/combo. second one reads those, applies cuts, writes a PDF.

## resample.py flags

```
--config path, default config.yaml
--era, only run one era
--sample, only run one background sample
--ext, override output filename tag
--run-id, write into an existing run dir instead of a new one
--suffix, appends _tag to the run id
```

run it multiple times with the same --run-id and different --era to build up one run instead of doing all eras at once.

## sculpting_check.py flags

```
--config path, default config.yaml
--run-id, which run to read, default is most recent
--era, only plot one era
--sample, only plot one background sample
--include-sample, keep only these (repeatable)
--exclude-sample, drop these (repeatable)
--weight-mode both/weighted/unweighted, default both
--cuts-file, use a different cuts file against the same run
--outdir, default <run_dir>/plots
--suffix, tag added to output PDF filename
--show, display instead of saving
```

## fold conventions

set under model.fold_convention in the config.

```
legacy - event % k_folds, offset routing
bhive_new - (event // 2) % k_folds, direct routing
detsplit_heldout - single model, only scores a held-out subset, needs split_k and held_out_folds too
```

use detsplit_heldout for a model with one checkpoint trained on part of the data. legacy/bhive_new will score the training half too, which is wrong.

## output layout

```
<output_base>/<run_id>/individual_samples/<era>/<sample>/y_resampled_<combo>_<ext>.parquet
<output_base>/<run_id>/plots/sculpting_<ext>.pdf
```

one PDF page per resample combo. one combo in the config means a one-page PDF.
