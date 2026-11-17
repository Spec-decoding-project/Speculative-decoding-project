"""Authenticate all source records and index prompts without exposing old labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

from feature_bank.data import records, family, source_name


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    counts, statuses, finishes, sources, lengths = Counter(), Counter(), Counter(), Counter(), []
    identities, ownership, largest = [], {}, []
    index = args.output/'prompts.jsonl'
    with index.open('w') as stream:
        for row in records(args.source):
            split = row['dataset_split']
            if split not in ('train', 'validation'):
                raise ValueError('Unknown source split')
            identity = family(row)
            if ownership.setdefault(identity, split) != split:
                raise ValueError('Cross-split normalized family overlap')
            prompt = row['prompt_token_ids']
            if not prompt or any(type(t) is not int or not 0 <= t < 151936 for t in prompt):
                raise ValueError('Invalid prompt token IDs')
            item = dict(id=row['id'], split=split, source=source_name(row), family_id=identity,
                        source_record_sha256=row['record_sha256'], prompt=prompt,
                        source_status=row['status'], source_finish_reason=row['finish_reason'])
            stream.write(json.dumps(item, separators=(',', ':'))+'\n')
            counts[split] += 1
            statuses[split+'/'+row['status']] += 1
            finishes[split+'/'+str(row['finish_reason'])] += 1
            sources[split+'/'+source_name(row)] += 1
            lengths.append(len(prompt))
            identities.append((row['id'], split, identity))
            largest.append((len(prompt), row['id'], split))
    if dict(counts) != {'train': 50000, 'validation': 1024}:
        raise ValueError(f'Unexpected source counts: {counts}')
    ordered = sorted(lengths)
    report = dict(counts=counts, statuses=statuses, finishes=finishes, sources=sources,
        prompt_tokens=dict(total=sum(lengths), mean=statistics.mean(lengths), min=min(lengths),
            max=max(lengths), quantiles={str(q):ordered[int(q*(len(ordered)-1))] for q in (.5,.9,.95,.99)},
            over1024=sum(n>1024 for n in lengths), over2240=sum(n>2240 for n in lengths),
            over6080=sum(n>6080 for n in lengths)),
        longest=sorted(largest, reverse=True)[:12], exact_normalized_family_overlap=0,
        source_hashes={p.name:sha(p) for p in [args.source/'run.json', *sorted(args.source.glob('shard-*.jsonl'))]},
        index_sha256=sha(index), elapsed_seconds=time.monotonic()-started)
    save(args.output/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
