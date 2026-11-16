"""Small synthetic compatibility check for an allocated Nexus CPU or GPU.

Uses no saved responses, checkpoints or target/draft models. Never a real-data fit.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import torch

from .learning import Trainer, evaluate
from .model import Config, TokenEditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new validation output path')
    torch.set_num_threads(1)
    config = Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32)
    rows = [dict(prompt_id=f'synthetic-{i}', anchor=3, prefix=[1, 2, 3 + i],
                 draft=[4, 5, 12, 13], labels=[4, 5, 6 + i, 8 + i]) for i in range(2)]
    result = dict(scope='Synthetic compatibility only; no real-data training or speedup evidence',
                  python=platform.python_version(), torch=str(torch.__version__), device=args.device,
                  hardware=torch.cuda.get_device_name() if args.device == 'cuda' else platform.machine(),
                  config=asdict(config), variants={})
    directory = Path(__file__).parent
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.py')}
    for mode in ('context', 'draft'):
        trainer = Trainer(config, mode, seed=3, lr=0.02, device=args.device)
        for _ in range(100):
            trainer.update(rows)
        metrics = evaluate(trainer.model, rows, mode)
        if metrics['teacher_forced_ce'] >= 0.05 or metrics['free_running_exact_suffix'] != 1.0:
            raise ValueError(f'{mode}: synthetic memorization check failed')
        for row in rows:
            cached = trainer.model.propose(row['prefix'], row['draft'], mode)
            eager = trainer.model.propose(row['prefix'], row['draft'], mode, cached=False)
            if cached != eager or cached != row['labels']:
                raise ValueError(f'{mode}: cached/free-running token mismatch')
        with tempfile.TemporaryDirectory(prefix='ar-synthetic-') as temporary:
            checkpoint = Path(temporary) / 'checkpoint.pt'
            trainer.save(checkpoint, {'origin': 'synthetic-test'})
            restored, _ = Trainer.load(checkpoint, device=args.device)
            for name, value in trainer.model.state_dict().items():
                if not torch.equal(value, restored.model.state_dict()[name]):
                    raise ValueError('Checkpoint weights changed')
            if evaluate(restored.model, rows, mode) != metrics:
                raise ValueError('Checkpoint predictions changed')
        result['variants'][mode] = dict(updates=100, teacher_forced_ce=metrics['teacher_forced_ce'],
                                        free_running_exact_suffix=metrics['free_running_exact_suffix'],
                                        cache_check='passed', checkpoint_check='passed')
    with torch.device('meta'):
        model = TokenEditor(Config())
    result['default_parameters'] = sum(p.numel() for p in model.parameters())
    result['outcome'] = 'passed'
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
