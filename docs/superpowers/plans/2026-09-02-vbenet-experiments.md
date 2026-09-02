# VBE-Net Experiment Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate VBE-Net into the repository's existing three-dataset experiment protocol without changing its architecture or evaluation rules.

**Architecture:** A `baselines.vbenet` experiment adapter instantiates the single-file `models.VBENet` and reuses the repository's established patch dataset, fixed-count split, metrics, and map writers. Thin CLI scripts expose train, train+eval, eval, and inference; generated YAML files mirror the current MGHOFNet experimental settings while replacing only model-specific keys.

**Tech Stack:** Python, PyTorch, PyYAML, unittest, Bash launchers.

**Spec:** `docs/superpowers/specs/2026-09-01-vbe-net-design.md`

## Fixed Alignment Contract

- Datasets: `yrd`, `yrd2509new`, `grss07`; shots: `5,10,20,50,100,150,200`.
- Fixed per-class train counts, no validation split, one final test after 200 epochs.
- Five seeds: `202201..202205`.
- Train batch 128; final test and labeled-map evaluation batch 1024; full-scene inference batch 512.
- AdamW with learning rate `1e-3`, weight decay `1e-2`; CosineAnnealingLR with `eta_min=1e-6`.
- Output path: `runs_fewshot/fs<shot>/vbenet/<dataset>/run_<NNN>`.
- Dataset-specific channels remain 10+4 for YRD/YRD2509NEW and 6+1 for GRSS07.
- Model defaults: width 64, depth 5, groups 8, expansion 4, lambda 1, `tau_r=.3`, `tau_c=.1`, inner iterations 3, outer updates 1, modality dropout .1.

### Task 1: Experiment Adapter

- [x] Write failing adapter tests for model construction, output directory, a finite training step, and config validation.
- [x] Implement `baselines/vbenet/train.py`, `eval.py`, and package exports using current shared data/metric/map infrastructure.
- [x] Verify adapter tests pass.

### Task 2: CLI and Config Grid

- [x] Write failing tests for the VBE config generator and experiment-grid inclusion.
- [x] Implement `train_vbenet.py`, `train_eval_vbenet.py`, `eval_vbenet.py`, `infer_vbenet.py`, and `generate_vbenet_configs.py`.
- [x] Generate 21 standard configs plus `vbenet_grss07_custom.yaml` by cloning the current training/evaluation settings and replacing model keys.
- [x] Add `vbenet` to the experiment grid and GRSS07 custom launcher.
- [x] Run script help, config validation, experiment dry-run, and the complete unit-test suite.

### Task 3: Commit

- [x] Commit only VBE experiment integration, generated configs, tests, and this plan.
