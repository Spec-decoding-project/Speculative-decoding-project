"""Build an immutable disk-backed token bank from independently re-audited chunks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from full_corpus_fit.collect import load_prompts, profile_prompts, read_chunk
from full_corpus_fit.data import audit_chunk
from full_corpus_fit.preflight import save, sha


class Bank:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(f'file:{self.path}?mode=ro', uri=True)
        self.metadata = json.loads(self.connection.execute('SELECT payload FROM metadata').fetchone()[0])

    def ids(self, split):
        return [r[0] for r in self.connection.execute('SELECT id FROM examples WHERE split=? ORDER BY id', (split,))]

    def row(self, identity):
        return json.loads(self.connection.execute('SELECT payload FROM examples WHERE id=?', (identity,)).fetchone()[0])

    def close(self):
        self.connection.close()


def build(preflight, collection, output):
    complete = json.loads((collection/'complete.json').read_text())
    run = json.loads((collection/'run.json').read_text())
    if sha(collection/'run.json') != complete['run_sha256'] or run['greedy']:
        raise ValueError('Require complete genuine native collection')
    prompts = load_prompts(preflight)
    if run['profile']:
        prompts = profile_prompts(prompts)
    mapping = {p['id']:p for p in prompts}
    if run['prompt_ids'] != [p['id'] for p in prompts] or len(prompts) != complete['prompts']:
        raise ValueError('Collection does not cover declared prompt inventory')
    if output.exists():
        old = Bank(output)
        expected = sha(collection/'complete.json')
        if old.metadata['collection_complete_sha256'] != expected:
            raise ValueError('Existing bank belongs to a different collection')
        old.close()
        return
    temporary = output.with_suffix('.tmp.sqlite')
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute('CREATE TABLE examples(id INTEGER PRIMARY KEY,split TEXT,prompt_id TEXT,kind TEXT,payload TEXT)')
    connection.execute('CREATE TABLE coverage(prompt_id TEXT PRIMARY KEY,split TEXT,payload TEXT)')
    connection.execute('CREATE TABLE metadata(payload TEXT)')
    seen, counts, covered, target_positions, natural = set(), Counter(), Counter(), 0, Counter()
    for name, expected in complete['chunks'].items():
        if sha(collection/name) != expected:
            raise ValueError('Collection chunk checksum mismatch')
        chunk = read_chunk(collection/name)
        batch = [mapping[identity] for identity in chunk['prompt_ids']]
        if chunk['run_sha256'] != complete['run_sha256']:
            raise ValueError('Chunk run provenance mismatch')
        fresh = audit_chunk(batch, chunk['generated'], chunk['observed'], hashes=run['profile'])
        if fresh != chunk['audit']:
            raise ValueError('Independent corpus chunk re-audit failed')
        for row in fresh['rows']:
            connection.execute('INSERT INTO examples(split,prompt_id,kind,payload) VALUES(?,?,?,?)',
                (row['split'], row['prompt_id'], row['kind'], json.dumps(row,separators=(',',':'))))
            counts[row['split']+'/'+row['kind']] += 1
        for row in fresh['coverage']:
            if row['prompt_id'] in seen:
                raise ValueError('Prompt occurs in multiple chunks')
            seen.add(row['prompt_id'])
            covered[row['split']+'/'+('used' if row['selected_blocks'] else 'no_eligible_block')] += 1
            connection.execute('INSERT INTO coverage VALUES(?,?,?)',
                (row['prompt_id'], row['split'], json.dumps(row)))
        target_positions += fresh['positions_checked']
        natural.update(fresh['natural_counts'])
        connection.commit()
    if seen != set(mapping):
        raise ValueError('Some authenticated prompts were never processed')
    metadata = dict(schema='full-corpus-token-bank-v1', collection_complete_sha256=sha(collection/'complete.json'),
        preflight_sha256=sha(preflight/'summary.json'), profile=run['profile'], prompt_count=len(seen),
        selected_counts=dict(counts), prompt_coverage=dict(covered), natural_counts=dict(natural),
        emitted_target_positions_verified=target_positions, editor_context_tokens=1024,
        target_contract='invariant-native-canonical-and-acceptance-v1', independent_greedy_scope='128 calibration prompts')
    connection.execute('INSERT INTO metadata VALUES(?)',(json.dumps(metadata),))
    connection.execute('CREATE INDEX split_index ON examples(split,id)')
    connection.commit()
    if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('SQLite bank integrity check failed')
    connection.close()
    temporary.replace(output)
    save(output.with_suffix('.json'), dict(**metadata, bank_sha256=sha(output), bytes=output.stat().st_size))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.preflight,args.collection,args.output)
    print(args.output.with_suffix('.json').read_text(),flush=True)


if __name__=='__main__':
    main()
