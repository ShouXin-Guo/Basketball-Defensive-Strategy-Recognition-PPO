"""Command line entry point for the basketball defensive strategy study."""
import argparse
import json
from pathlib import Path
from config import load_config, list_configurations
from data import load_events
from experiment import run, select_device, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train','optimize','evaluate','list-configs'], default='train')
    parser.add_argument('--data', help='Annotated CSV or NPZ with features, labels and game identifiers')
    parser.add_argument('--config', default='trained_configurations/initial.json')
    parser.add_argument('--config-dir', default='trained_configurations')
    parser.add_argument('--output', default='outputs/run')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--checkpoint', help='Saved .pt checkpoint (evaluate mode)')
    parser.add_argument('--five-seeds', action='store_true', help='Use manuscript seeds 42,123,2024,3407,2026')
    args = parser.parse_args()
    if args.mode == 'list-configs':
        print(json.dumps({name: c.status for name, c in list_configurations(args.config_dir).items()}, indent=2))
        return
    if not args.data:
        parser.error('--data is required except for list-configs')
    data = load_events(args.data)
    device = select_device(args.device)
    if args.mode == 'evaluate':
        if not args.checkpoint:
            parser.error('--checkpoint is required for evaluate')
        result = evaluate_checkpoint(data, args.checkpoint, device, args.output)
    else:
        cfg = load_config(args.config)
        if args.mode == 'train' and cfg.status == 'trained':
            raise ValueError('Reuse a trained configuration for optimization/training only after confirming the intended experiment')
        seeds = [42,123,2024,3407,2026] if args.five_seeds else [cfg.seed]
        result = run(data, cfg, device, args.output, optimize_first=(args.mode=='optimize'), seeds=seeds)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
