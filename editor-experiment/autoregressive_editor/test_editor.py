"""Synthetic checks of real model, training, data and CLI paths; no model downloads."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from autoregressive_editor.data import Bank, checksum, condition, teacher_forcing
from autoregressive_editor.learning import Trainer, evaluate, tensors
from autoregressive_editor.model import Config, TokenEditor

torch.set_num_threads(1)


def example(index=0):
    return dict(prompt_id=f'p{index}', family_id=f'f{index}', split='train',
                source_record_sha256='synthetic-fixture', anchor=3, prefix=[1, 2, 3 + index],
                draft=[4, 5, 12, 13], labels=[4, 5, 6 + index, 8 + index], target_replay_verified=True)


def config():
    return Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32)


def write_bank(path, examples):
    path.mkdir()
    rows = path / 'examples.jsonl'
    rows.write_text(''.join(json.dumps(r) + '\n' for r in examples))
    (path / 'manifest.json').write_text(json.dumps(dict(
        schema='autoregressive-token-bank-v1', outcome='validated', origin='synthetic-test',
        target_replay='passed', vocab_size=20, models={'fixture': 'synthetic'},
        examples=len(examples), examples_sha256=checksum(rows))))
    return path


def test_conditioning_is_the_only_input_difference():
    e = example()
    altered = deepcopy(e)
    altered['draft'][2:] = [17, 18]
    assert condition(e['prefix'], e['draft'], 'context') == condition(altered['prefix'], altered['draft'], 'context')
    assert condition(e['prefix'], e['draft'], 'draft') != condition(altered['prefix'], altered['draft'], 'draft')
    for mode in ('context', 'draft'):
        ids, _, start, labels = teacher_forcing(e, mode)
        base, _ = condition(e['prefix'], e['draft'], mode)
        assert ids[:len(base)] == base
        assert start == len(base) - 1 and labels == [6, 8]
        assert ids[start:] == [5, 6]  # Predict 6 from seed 5; predict 8 from correct 6.


@pytest.mark.parametrize('mode', ['context', 'draft'])
def test_future_corrected_tokens_cannot_change_earlier_prediction(mode):
    trainer = Trainer(config(), mode)
    a, b = example(), example()
    b['labels'][2] = 19
    ia, sa, start, _ = tensors(trainer.model, a, mode)
    ib, sb, _, _ = tensors(trainer.model, b, mode)
    la, _ = trainer.model(ia, sa, logit_start=start)
    lb, _ = trainer.model(ib, sb, logit_start=start)
    torch.testing.assert_close(la[:, 0], lb[:, 0], atol=0, rtol=0)


def test_cached_logits_match_full_forward_and_reset_between_rounds():
    torch.manual_seed(4)
    model = TokenEditor(config()).double().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 2, 2]])
    full, _ = model(ids, segments)
    first, cache = model(ids[:, :4], segments[:, :4], use_cache=True)
    outputs = [first]
    for position in range(4, ids.shape[1]):
        value, cache = model(ids[:, position:position + 1], segments[:, position:position + 1], past=cache, use_cache=True)
        outputs.append(value)
    torch.testing.assert_close(full, torch.cat(outputs, dim=1), atol=1e-12, rtol=1e-12)
    for mode in ('context', 'draft'):
        e = example()
        expected = model.propose(e['prefix'], e['draft'], mode, cached=False)
        assert model.propose(e['prefix'], e['draft'], mode) == expected
        model.propose([9, 10], [4, 5, 11], mode)
        assert model.propose(e['prefix'], e['draft'], mode) == expected
        assert expected[:2] == e['draft'][:2]


@pytest.mark.parametrize('mode', ['context', 'draft'])
def test_real_training_path_can_memorize_two_synthetic_blocks(mode):
    trainer = Trainer(config(), mode, seed=3, lr=0.02)
    rows = [example(0), example(1)]
    initial = evaluate(trainer.model, rows, mode)
    for _ in range(100):
        trainer.update(rows)
    result = evaluate(trainer.model, rows, mode)
    assert result['teacher_forced_ce'] < 0.05
    assert result['teacher_forced_ce'] < initial['teacher_forced_ce']
    assert result['free_running_exact_suffix'] == 1.0
    assert result['damaged'] == 0


def test_checkpoint_resumes_optimizer_and_example_order_exactly(tmp_path):
    trainer = Trainer(config(), 'draft', seed=4, lr=0.01)
    rows = [example(0), example(1)]
    for _ in range(3):
        trainer.update([rows[trainer.rng.randrange(2)]])
    path = tmp_path / 'checkpoint.pt'
    trainer.save(path, {'bank': 'synthetic'})
    restored, metadata = Trainer.load(path)
    assert metadata == {'bank': 'synthetic'} and restored.step == trainer.step
    for _ in range(3):
        a, b = trainer.rng.randrange(2), restored.rng.randrange(2)
        assert a == b
        trainer.update([rows[a]])
        restored.update([rows[b]])
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)


def test_bank_rejects_corruption_unverified_replay_and_split_leakage(tmp_path):
    good = write_bank(tmp_path / 'good', [example()])
    assert len(Bank(good)) == 1
    (good / 'examples.jsonl').write_text('corrupted\n')
    with pytest.raises(ValueError, match='checksum'):
        Bank(good)
    row = example()
    row['target_replay_verified'] = False
    with pytest.raises(ValueError, match='replay'):
        Bank(write_bank(tmp_path / 'unverified', [row]))
    a, b = example(), example(1)
    b['family_id'], b['split'] = a['family_id'], 'validation'
    with pytest.raises(ValueError, match='leakage'):
        Bank(write_bank(tmp_path / 'mixed', [a, b]))
    a, b = example(), example()
    b['split'] = 'validation'
    train = Bank(write_bank(tmp_path / 'train', [a]))
    valid = Bank(write_bank(tmp_path / 'valid', [b]))
    with pytest.raises(ValueError, match='overlap'):
        train.require_disjoint(valid)


def test_wrong_first_proposal_is_counted_at_evaluation_and_rejected_for_fit():
    trainer = Trainer(config(), 'draft')
    row = example()
    row['draft'][1] = 19
    with pytest.raises(ValueError, match='reachable'):
        trainer.update([row])
    result = evaluate(trainer.model, [row], 'draft')
    assert result['wrong_first'] == 1 and result['repaired'] == 0
    assert result['rows'][0]['union_progress'] == 0


def test_cli_trains_synthetic_bank_and_reloads(tmp_path):
    path = write_bank(tmp_path / 'bank', [example()])
    experiment = tmp_path / 'experiment.json'
    experiment.write_text(json.dumps(dict(name='synthetic', conditioning='context', seed=1,
                                          learning_rate=0.01, model=asdict(config()))))
    cwd = Path(__file__).resolve().parents[1]
    base = [sys.executable, '-m', 'autoregressive_editor.run']
    arguments = ['--experiment', str(experiment), '--bank', str(path)]
    subprocess.run(base + ['train'] + arguments + ['--output', str(tmp_path / 'run'), '--updates', '2',
                   '--batch-size', '1', '--eval-every', '1', '--threads', '1'], cwd=cwd, check=True, capture_output=True)
    subprocess.run(base + ['evaluate'] + arguments + ['--checkpoint', str(tmp_path / 'run/checkpoint.pt'),
                   '--output', str(tmp_path / 'eval')], cwd=cwd, check=True, capture_output=True)
    summary = json.loads((tmp_path / 'run/summary.json').read_text())
    assert summary['updates'] == 2
    assert json.loads((tmp_path / 'eval/evaluation.json').read_text())['blocks'] == 1


def test_export_requires_full_gate_and_authenticates_source_records(tmp_path):
    from safetensors.torch import save_file
    from feature_bank.data import digest
    from autoregressive_editor.export_bank import export
    source, responses = tmp_path / 'source', tmp_path / 'responses'
    source.mkdir()
    responses.mkdir()
    (source / 'audit.json').write_text(json.dumps({'outcome': 'failed'}))
    with pytest.raises(ValueError, match='partial banks'):
        export(source, responses, tmp_path / 'rejected')
    run = {'fixture': 'synthetic'}
    (responses / 'run.json').write_text(json.dumps(run))
    blocks, source_rows = [], []
    models = json.loads((Path(__file__).resolve().parents[1] / 'feature_bank/model_pin.json').read_text())
    for index in range(144):
        labels = [4, 5] + [6] * 13 + [151645]
        draft = list(labels)
        if index < 128:
            draft[2] = 19
        row = dict(id=f'prompt-{index}', dataset_split='train', status='generated', finish_reason='eos',
                   prompt_token_ids=[1, 2, 3], response_token_ids=labels,
                   messages=[{'role': 'user', 'content': f'Synthetic {index}'}], run_sha256=digest(run))
        row['record_sha256'] = digest(row)
        source_rows.append(row)
        name = f'block-{index:04d}.safetensors'
        save_file(dict(normalized_states=torch.zeros(1, 16, 4096, dtype=torch.bfloat16),
                       first_guess_embedding=torch.zeros(1, 4096, dtype=torch.bfloat16),
                       original_ids=torch.tensor([draft]), labels=torch.tensor([labels]),
                       valid_mask=torch.ones(1, 16, dtype=torch.bool), prefix_ids=torch.tensor([[1, 2, 3]]),
                       anchor=torch.tensor([3])), source / name)
        blocks.append(dict(id=row['id'], dataset_split='train', anchor=3, file=name,
                           kind='repairable' if index < 128 else 'preservation',
                           original_ids=draft, labels=labels, record_sha256=row['record_sha256'],
                           sha256=checksum(source / name), bytes=(source / name).stat().st_size))
    (responses / 'shard-00000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in source_rows))
    (source / 'manifest.json').write_text(json.dumps(dict(schema_version=1, models=models, blocks=blocks)))
    (source / 'audit.json').write_text(json.dumps(dict(outcome='success', manifest_sha256=checksum(source / 'manifest.json'))))
    bank = export(source, responses, tmp_path / 'tokens')
    assert len(bank) == 144 and bank[0]['prefix'] == [1, 2, 3]
    # A validly checksummed response journal is still wrong if it differs from the bank's source.
    source_rows[0]['response_token_ids'][2] = 7
    source_rows[0]['record_sha256'] = digest({k: v for k, v in source_rows[0].items() if k != 'record_sha256'})
    (responses / 'shard-00000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in source_rows))
    with pytest.raises(ValueError, match='provenance'):
        export(source, responses, tmp_path / 'wrong-source')
