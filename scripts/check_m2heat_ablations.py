"""Synthetic forward/backward and GPU peak profiling; never load dataset arrays."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from baselines.m2heat.ablation import VARIANTS, MODULE_VARIANTS, M2HeatAblation
    from baselines.training import set_seed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--suite', choices=('legacy', 'modules'), default='legacy')
    parser.add_argument('--profile', action='store_true', help='Use protocol train128 and eval1024 batches')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.init()
    records = []
    for ms, sar, classes in ((10, 4, 8), (10, 4, 9), (6, 1, 5)):
        for variant in MODULE_VARIANTS if args.suite == 'modules' else VARIANTS:
            set_seed(202201)
            if device.type == 'cuda':
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            model = M2HeatAblation(variant, num_classes=classes, spectral_channels=ms, sar_channels=sar).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            size = 128 if args.profile else 2
            x = torch.randn(size, 1, ms + sar, 11, 11, device=device)
            y = torch.arange(size, device=device) % classes
            output = model(x)
            assert output.shape == (size, classes) and torch.isfinite(output).all()
            torch.nn.functional.cross_entropy(output, y).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            del x, y, output
            model.eval()
            with torch.no_grad():
                x = torch.randn(1024 if args.profile else 2, 1, ms + sar, 11, 11, device=device)
                output = model(x)
                assert torch.isfinite(output).all()
            record = dict(variant=variant, channels=[ms, sar], classes=classes, device=str(device), passed=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                record.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
            records.append(record)
            print(json.dumps(record), flush=True)
            del model, optimizer, x, output
            gc.collect()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
