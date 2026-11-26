"""Frozen COPY/vocabulary counterfactuals; no optimizer or target-model calls."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import torch

from autoregressive_editor.data import condition
from autoregressive_editor.model import Config, TokenEditor
from copy_action_fit.analyze import analyze
from copy_action_fit.data import read_bank, save, sha
from copy_action_fit.learning import aggregate, evaluate, weights_hash
from copy_action_fit.model import CopyEditor
from copy_action_fit.train import audit, require, uncertainty


HASHES = dict(initial='c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2',
    baseline='5b03244e1a28db6aaacd8ba8d517ca2c8f7f192da894999ae8d6f9bbcc01d0d9',
    copy='bcde53a197f3152c3f9cc9b1e3c3d49ed5a715635e3ce4027b3338bb0403ab39')


def skipped_cases(control, copied):
    require(len(control['rows']) == len(copied['rows']), 'Population mismatch')
    cases = []
    for index, (left, right) in enumerate(zip(control['rows'], copied['rows'])):
        for key in ('prompt_id', 'anchor', 'original_tokens', 'label_tokens'):
            require(left[key] == right[key], 'Unpaired prediction rows')
        if left['repaired'] and right['edited_progress'] == right['original_progress']:
            position = right['original_progress']+1
            require(2 <= position < 16, 'Repair lies outside editable positions')
            if right['actions'][position-2] == 151936:
                require(right['edited_tokens'][:position] == right['label_tokens'][:position],
                        'Counterfactual preceding prefix is not the saved correct prefix')
                require(left['edited_tokens'][:position+1] == left['label_tokens'][:position+1],
                        'Control does not actually repair the first error')
                cases.append(dict(index=index, position=position, trace=right['edited_tokens']))
    return cases


@torch.no_grad()
def probe(model, example, trace, position):
    """Force only the recorded past; never include the current reference token."""
    ids, segments = condition(example['prefix'], example['draft'], 'draft')
    device = model.token.weight.device
    tokens = torch.tensor([ids], device=device)
    types = torch.tensor([segments], device=device)
    past = None
    for current in range(2, position+1):
        logits, past = model(tokens, types, past=past, use_cache=True,
                             logit_start=tokens.shape[1]-1)
        if current < position:
            tokens = torch.tensor([[trace[current]]], device=device)
            types = torch.full_like(tokens, 2)
    scores = logits[0, -1].float()
    vocabulary = scores[:model.config.vocab_size]
    reference = example['labels'][position]
    top_values, top_ids = vocabulary.topk(2)
    competitor = top_values[1] if int(top_ids[0]) == reference else top_values[0]
    result = dict(vocabulary_argmax=int(vocabulary.argmax()), action_argmax=int(scores.argmax()),
        reference_token=reference, original_token=example['draft'][position],
        reference_rank=int((vocabulary > vocabulary[reference]).sum())+1,
        reference_margin=float(vocabulary[reference]-competitor))
    if scores.numel() > model.config.vocab_size:
        result.update(copy_over_best_vocabulary=float(scores[-1]-vocabulary.max()),
                      copy_over_reference=float(scores[-1]-vocabulary[reference]))
    return result


def load_model(path, variant, device='cuda'):
    require(sha(path) == HASHES[variant], 'Frozen checkpoint fingerprint changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    config = Config(**payload['config'])
    model = CopyEditor(config) if variant == 'copy' else TokenEditor(config)
    model.load_state_dict(payload['model'], strict=True)
    del payload
    return model.requires_grad_(False).to(device).eval()


def vocabulary_only(model):
    plain = TokenEditor(model.config)
    plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('copy_head.')})
    require(weights_hash(plain) == weights_hash(model, shared=True), 'Ablation changed shared weights')
    return plain.requires_grad_(False).to(model.token.weight.device).eval()


def run(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    train, dev, selection = read_bank(args.bank)
    control = json.loads((args.pilot/'baseline/development.json').read_text())
    copied = json.loads((args.pilot/'copy/development.json').read_text())
    cases = skipped_cases(control, copied)
    require(len(cases) == 231, 'Skipped-repair population changed')
    save(args.output/'action-partition.json', analyze(control, copied))
    save(args.output/'run.json', dict(selection=selection, checkpoint_sha256=HASHES,
         runner_sha256=sha(Path(__file__)), optimizer_updates=0, final_test_loaded=False,
         torch=str(torch.__version__), gpu=torch.cuda.get_device_name()))
    probes, replays = {}, {}
    paths = dict(initial=args.initial, baseline=args.pilot/'baseline/checkpoint.pt',
                 copy=args.pilot/'copy/checkpoint.pt')
    for variant in ('initial', 'baseline', 'copy'):
        print(json.dumps(dict(phase='replay_and_probe', variant=variant)), flush=True)
        model = load_model(paths[variant], variant)
        before = weights_hash(model)
        mode = 'copy' if variant == 'copy' else 'baseline'
        replay = evaluate(model, dev, mode)
        reference = (json.loads((args.pilot/'baseline/initial-development.json').read_text())
                     if variant == 'initial' else control if variant == 'baseline' else copied)
        require(replay == reference, f'{variant} development replay changed')
        replays[variant] = dict(passed=True, blocks=len(dev), weights_sha256=before)
        probes[variant] = []
        for case in cases:
            row = dev[case['index']]
            result = probe(model, row, case['trace'], case['position'])
            if variant == 'copy':
                require(result['action_argmax'] == model.config.vocab_size,
                        'Recorded COPY action no longer reproduced on the identical prefix')
            elif variant == 'baseline':
                require(result['vocabulary_argmax'] == result['reference_token'],
                        'Recorded control repair not reproduced on the identical prefix')
            probes[variant].append(dict(prompt_id=row['prompt_id'], anchor=row['anchor'],
                                        position=case['position'], **result))
        if variant in ('baseline', 'copy'):
            training = evaluate(model, train, mode)
            require(training == json.loads((args.pilot/variant/'final-training.json').read_text()),
                    'Training checkpoint replay changed')
        if variant == 'copy':
            plain = vocabulary_only(model)
            ablation = {}
            for split, rows in (('training', train), ('development', dev)):
                result = evaluate(plain, rows, 'baseline')
                save(args.output/f'vocabulary-only-{split}.json', result)
                save(args.output/f'vocabulary-only-{split}-audit.json', audit(result, rows, 'baseline', 151936))
                ablation[split] = aggregate(result, rows)
            save(args.output/'vocabulary-only-summary.json', ablation)
            for row in dev[::32]:
                require(plain.propose(row['prefix'], row['draft'], 'draft') ==
                        plain.propose(row['prefix'], row['draft'], 'draft', cached=False),
                        'Vocabulary-only cache/full mismatch')
            del plain
        require(weights_hash(model) == before, 'Frozen model weights changed')
        del model
        gc.collect(); torch.cuda.empty_cache()
    save(args.output/'probes.json', probes)
    counts = Counter()
    for initial, control_probe, copied_probe in zip(probes['initial'], probes['baseline'], probes['copy']):
        reference = copied_probe['reference_token']
        initial_correct = initial['vocabulary_argmax'] == reference
        copy_correct = copied_probe['vocabulary_argmax'] == reference
        counts['cases'] += 1
        counts['copy_overrules_correct_vocabulary' if copy_correct else 'copy_vocabulary_incorrect'] += 1
        counts['starting_vocabulary_correct' if initial_correct else 'starting_vocabulary_incorrect'] += 1
        if initial_correct and not copy_correct:
            counts['correct_starting_prediction_lost'] += 1
    disabled = json.loads((args.output/'vocabulary-only-development.json').read_text())
    original = [dict(r, edited_progress=r['original_progress']) for r in disabled['rows']]
    intervals = dict(disabled_minus_original=uncertainty(disabled['rows'], original),
                     disabled_minus_copy=uncertainty(disabled['rows'], copied['rows']))
    require(all(sha(path) == HASHES[variant] for variant,path in paths.items()), 'Input checkpoint changed')
    complete = dict(passed=True, diagnostic_counts=counts, replays=replays,
        vocabulary_only=ablation, uncertainty=intervals, elapsed_seconds=time.monotonic()-started,
        optimizer_updates=0, final_test_loaded=False,
        scope='Frozen counterfactual diagnosis; oracle error positions are not an inference policy')
    save(args.output/'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'pilot', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
