"""Train only a requested M2Heat ablation, independently of the original entry."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--seed', type=int, choices=range(202201, 202206))
    parser.add_argument('--variant', help='Module variant to apply to the original dataset configuration')
    args = parser.parse_args()
    from baselines.m2heat.ablation_experiment import train
    print(train(args.config, args.batch_dir, args.seed, variant=args.variant), flush=True)


if __name__ == '__main__':
    main()
