# MSI-SAR Baselines

This repository keeps datasets and comparison methods for multispectral-SAR
classification experiments. Each baseline lives in its own folder.

## Layout

```text
baselines/
  dfinet/              DFI-Net comparison method
  frekfuse/            FreKFuse comparison method
  mghofnet/            MG-HOFNet comparison method
  msfmamba/            MSFMamba comparison method
  softformer/          SoftFormer comparison method
configs/
  datasets/yrd.yaml    Dataset description
  datasets/grss07.yaml GRSS-DFC-2007 dataset description
  *_yrd_fs{5,10,20,50,100}.yaml  Fixed-count few-shot experiment configs
data/
  yrd/data.mat
  yrd/label.mat
  grss07/data.mat       6 Landsat channels followed by 1 ERS channel
  grss07/label.mat      Labels 0-5; 0 is undefined
runs/                  Experiment outputs
runs_fewshot/fs5/      5-shot experiment outputs
runs_fewshot/fs10/     10-shot experiment outputs
runs_fewshot/fs20/     20-shot experiment outputs
runs_fewshot/fs50/     50-shot experiment outputs
runs_fewshot/fs100/    100-shot experiment outputs
scripts/               Thin command-line wrappers
vendor/                Original third-party code snapshots
```

## Commands

```bash
python scripts/prepare_grss07.py \
  --source /path/to/GRSS07_SAR_MS.mat \
  --output data/grss07
python scripts/generate_grss07_configs.py
python scripts/check_dataset.py --dataset grss07

python scripts/check_dataset.py --dataset yrd
python scripts/train_dfinet.py --config configs/dfinet_yrd_fs5.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs5.yaml
python scripts/eval_dfinet.py --run runs_fewshot/fs5/dfinet/yrd/run_001
python scripts/infer_dfinet.py --run runs_fewshot/fs5/dfinet/yrd/run_001

python scripts/train_frekfuse.py --config configs/frekfuse_yrd_fs5.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs5.yaml
python scripts/eval_frekfuse.py --run runs_fewshot/fs5/frekfuse/yrd/run_001
python scripts/infer_frekfuse.py --run runs_fewshot/fs5/frekfuse/yrd/run_001

python scripts/train_mghofnet.py --config configs/mghofnet_yrd_fs5.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs5.yaml
python scripts/eval_mghofnet.py --run runs_fewshot/fs5/mghofnet/yrd/run_001
python scripts/infer_mghofnet.py --run runs_fewshot/fs5/mghofnet/yrd/run_001

python scripts/train_msfmamba.py --config configs/msfmamba_yrd_fs5.yaml
python scripts/train_eval_msfmamba.py --config configs/msfmamba_yrd_fs5.yaml
python scripts/eval_msfmamba.py --run runs_fewshot/fs5/msfmamba/yrd/run_001
python scripts/infer_msfmamba.py --run runs_fewshot/fs5/msfmamba/yrd/run_001

python scripts/train_softformer.py --config configs/softformer_yrd_fs5.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs5.yaml
python scripts/eval_softformer.py --run runs_fewshot/fs5/softformer/yrd/run_001
python scripts/infer_softformer.py --run runs_fewshot/fs5/softformer/yrd/run_001
```

## GRSS-DFC-2007 Experiments

The NCGLF2 subset uses the official 1994 Landsat image (six optical bands),
the ERS acquisition from 1994-10-03 (one SAR band), and the five-class ground
truth. The conversion script concatenates these arrays without changing their
pixel values. Configs cover all five baselines and the `fs5`, `fs10`, `fs20`,
`fs50`, `fs100`, `fs150`, and `fs200` settings.

Run the complete GRSS07 grid in the `gjc` environment with one GPU and two
concurrent jobs by default:

```bash
bash scripts/run_grss07_experiments.sh
```

Preview the queue or override concurrency:

```bash
DRY_RUN=1 bash scripts/run_grss07_experiments.sh
GPU_COUNT=1 PROCS_PER_GPU=1 bash scripts/run_grss07_experiments.sh
```

## Few-Shot Commands

Few-shot configs use fixed per-class train/validation counts. Results are saved
under one folder per shot setting.

```bash
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs5.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs10.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs20.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs50.yaml
python scripts/train_eval_dfinet.py --config configs/dfinet_yrd_fs100.yaml
```

```bash
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs5.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs10.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs20.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs50.yaml
python scripts/train_eval_frekfuse.py --config configs/frekfuse_yrd_fs100.yaml
```

```bash
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs5.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs10.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs20.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs50.yaml
python scripts/train_eval_mghofnet.py --config configs/mghofnet_yrd_fs100.yaml
```

```bash
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs5.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs10.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs20.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs50.yaml
python scripts/train_eval_softformer.py --config configs/softformer_yrd_fs100.yaml
```

Few-shot settings:

```text
fs5:  train/class=5,  val/class=5,  train total=40,  val total=40
fs10: train/class=10, val/class=10, train total=80,  val total=80
fs20: train/class=20, val/class=20, train total=160, val total=160
fs50: train/class=50, val/class=50, train total=400, val total=400
fs100: train/class=100, val/class=100, train total=800, val total=800
```

DFI-Net accepts the `patch_size` set in `configs/dfinet_yrd_fs5.yaml`. Training
tests only labeled test samples by default; run `infer_dfinet.py` only when a
full-scene map is needed. The default YRD configuration is 5-shot: five
training and five validation pixels per class. Select another shot config as
needed.

`train_eval_dfinet.py` also saves labeled-sample maps for the selected run:

```text
labeled_gt.tif
labeled_gt.png
labeled_prediction.tif
labeled_prediction.png
```

`infer_dfinet.py` saves full-scene prediction maps:

```text
full_prediction.tif
full_prediction.png
```

MGHOFNet is adapted from the previous HSI-LiDAR model under
`D:\gaojianchun\fenlei\ttt`; the original files are preserved in
`vendor/MGHOFNet-original/`. In `configs/mghofnet_yrd_fs5.yaml`, YRD uses the first 10
channels as the MSI/HSI branch and the last 4 channels as the SAR/auxiliary
branch.

SoftFormer is adapted from `rl1024/SoftFormer`; the original files are
preserved in `vendor/SoftFormer-original/`. In `configs/softformer_yrd_fs5.yaml`,
YRD uses the first 10 channels as the optical/MSI branch and the last 4 channels
as the SAR branch.

FreKFuse uses the first 10 YRD channels as MSI and the last 4 channels as SAR.
The full-data variant uses embed_dim=256 and spline_order=3, while few-shot
configs automatically switch to FreKFuseLite (embed_dim=128, spline_order=2,
dropout=0.5) for better regularization. A patch size of 32 is required for
meaningful frequency-domain analysis.

MSFMamba is adapted from the official [oucailab/MSFMamba](https://github.com/oucailab/MSFMamba)
release (IEEE TGRS 2025). Its original source is stored in
`vendor/MSFMamba-original/`; follow `baselines/msfmamba/README.md` to install
its pinned runtime without allowing pip to upgrade the project's Torch wheel.
The 10 MSI channels feed its spectral branch and the remaining
2/4 channels feed its SAR branch. Configs for YRD and YRD2509NEW are
generated reproducibly with `python scripts/generate_msfmamba_configs.py`.

GRSS07 has only six optical bands. For this dataset, the MSFMamba adapter
zero-pads the spectral axis to nine bands before the released fixed 9-band
spectral convolution. This adds no synthetic observation values and leaves the
rest of the official layer unchanged.
