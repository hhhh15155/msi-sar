# YRD2509NEW few-shot experiments

The experiment grid matches the existing YRD/YRD2509 benchmark:

- models: `dfinet`, `frekfuse`, `mghofnet`, `softformer`
- shots: `5, 10, 20, 50, 100, 150, 200`
- repetitions: 5 seeds
- input: 10 optical channels + 4 YRD-matched SAR channels
- outputs: `runs_fewshot/fs{shot}/{model}/yrd2509new/run_###`

Run only YRD2509NEW sequentially:

```bash
python scripts/train_eval_yrd2509new_fewshot.py
```

Select models and shots:

```bash
python scripts/train_eval_yrd2509new_fewshot.py \
  --models frekfuse mghofnet \
  --shots 100 150
```

Run the complete GPU scheduler:

```bash
bash scripts/run_all_experiments.sh
```

The scheduler skips any experiment whose `run_001/metrics.json` already
exists. Existing YRD/YRD2509 results are therefore not rerun.

## Bare-soil support

Class 7 contains 571 trusted pixels after expanding its continuous land-use
interior core. The 200-shot protocol therefore retains 200 training, 200
validation, and 171 test pixels for this class.
