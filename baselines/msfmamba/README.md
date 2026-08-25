# MSFMamba Baseline

This adapter integrates the official [MSFMamba](https://github.com/oucailab/MSFMamba)
implementation (Gao *et al.*, IEEE TGRS 2025) into this project's deterministic
MSI-SAR experiment interface. The untouched author snapshot is in
`vendor/MSFMamba-original/`; the adapter supplies the project dataset reader,
fixed-count split, metrics, checkpoint, and map-export conventions.

The YRD datasets have ten MSI bands, so the released 11x11, one-layer network
uses them directly as its spectral input (PCA is neither necessary nor possible
when the input only has ten bands). The remaining 2 or 4 bands form the SAR
input. This is an input adaptation, not a structural change to the official
`Syn_layer` and Mamba blocks.

For the paper's original software stack, the authors list
`causal-conv1d==1.1.1` and `mamba-ssm==1.0.1`. For current Blackwell GPUs and
PyTorch 2.11, use the newer compatible packages in this project instead. The
CUDA toolkit used to compile the extensions must match `torch.version.cuda`.
For example, a PyTorch `+cu128` environment needs CUDA Toolkit 12.8 (the
driver version is not the compiler version).

Install the optional upstream runtime once. First install the torchvision wheel
paired with the chosen PyTorch wheel, then install Mamba and timm without
letting pip resolve a newer Torch. For the project's RTX 5090 / Torch
`2.11.0+cu128` environment:

```bash
pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.36,<5"
pip install --no-deps --no-build-isolation mamba-ssm==2.2.6.post3
pip install --no-deps timm==0.6.13
```

Before the Mamba command, set `CUDA_HOME` to a CUDA 12.8 Toolkit installation
and put `$CUDA_HOME/bin` at the beginning of `PATH`. Do not combine these three
packages into one unconstrained `pip install` command.

Then use the same train/evaluate/infer workflow as every other baseline:

```bash
python scripts/train_eval_msfmamba.py --config configs/msfmamba_yrd2509_fs5.yaml
python scripts/eval_msfmamba.py --run runs_fewshot/fs5/msfmamba/yrd2509/run_001
python scripts/infer_msfmamba.py --run runs_fewshot/fs5/msfmamba/yrd2509/run_001
```
