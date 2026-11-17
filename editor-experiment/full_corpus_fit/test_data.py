import unittest

from full_corpus_fit.data import audit_chunk, token_hash, split_prompts, KNOWN_DEVELOPMENT


def fixture(length=10, wrong_first=False):
    prompts, generated, blocks, trace = [], [], [], []
    # Two requests have identical prompt tokens: identity must not be guessed
    # from prompt content, and request-local evidence must remain separate.
    for i in range(2):
        prompt = [100]*length
        labels = list(range(20,52))
        draft = labels[:16]
        draft[1 if wrong_first else 5] = 99
        accepted = 0 if wrong_first else 4
        internal = str(i)+'-abcdef01'
        prompts.append(dict(id=f'p{i}',prompt=prompt,split='train',source='fixture',
            family_id='same-family',source_record_sha256='fixture'))
        generated.append(dict(request_id=str(i),prompt=list(prompt),tokens=labels))
        blocks.append(dict(request_id=internal,prefix=list(prompt),anchor=length,draft=draft,
            positions=list(range(length,length+16)),native_draft_count=15,
            num_sampled=accepted+1,num_rejected=15-accepted))
        for j, token in enumerate(labels):
            trace.append(dict(request_id=internal,position=length+j-1,
                prefix_sha256=token_hash(prompt+labels[:j]),argmax=token,vocabulary_size=151936))
    return prompts, generated, dict(trace=trace,blocks=blocks)


class CorpusIntegrity(unittest.TestCase):
    def test_identical_prompts_are_mapped_by_request(self):
        a,b,c=fixture()
        result=audit_chunk(a,b,c)
        self.assertEqual(result['positions_checked'],64)
        self.assertEqual([r['prompt_id'] for r in result['rows']],['p0','p1'])

    def test_long_context_is_windowed_only_after_native_audit(self):
        a,b,c=fixture(1500)
        row=audit_chunk(a,b,c)['rows'][0]
        self.assertEqual((len(row['prefix']),row['context_start'],row['anchor']),(1024,476,1500))
        c['blocks'][0]['prefix'][0]=99
        with self.assertRaisesRegex(ValueError,'context'):
            audit_chunk(a,b,c)

    def test_native_acceptance_and_reference_argmax_corruption_fail(self):
        a,b,c=fixture()
        c['blocks'][0]['num_sampled']=1
        with self.assertRaisesRegex(ValueError,'acceptance'):
            audit_chunk(a,b,c)
        a,b,c=fixture()
        c['trace'][0]['argmax']=1
        with self.assertRaisesRegex(ValueError,'argmax'):
            audit_chunk(a,b,c)

    def test_wrong_first_is_logged_not_used_for_training(self):
        a,b,c=fixture(wrong_first=True)
        result=audit_chunk(a,b,c)
        self.assertFalse(result['rows'])
        self.assertEqual(result['coverage'][0]['exclusion_reason'],'no_reachable_full_block')
        for prompt in a:
            prompt['split']='test'
        self.assertEqual(len(audit_chunk(a,b,c)['rows']),2)

    def test_already_correct_native_blocks_are_included(self):
        a,b,c=fixture()
        for block in c['blocks']:
            block['draft']=list(range(20,36))
            block.update(num_sampled=16,num_rejected=0)
        result=audit_chunk(a,b,c)
        self.assertTrue(all(r['kind']=='preservation' for r in result['rows']))

    def test_previously_inspected_prompts_cannot_enter_final_test(self):
        rows=[]
        for source in ('gsm8k','ultrachat'):
            known=sorted(k for k in KNOWN_DEVELOPMENT if k.startswith('gsm8k')==(source=='gsm8k'))
            names=known+[f'{source}:fresh:{i}' for i in range(512-len(known))]
            rows.extend(dict(id=name,split='validation',source=source) for name in names)
        split=split_prompts(rows)
        self.assertEqual(sum(r['split']=='development' for r in split),512)
        self.assertEqual(sum(r['split']=='test' for r in split),512)
        self.assertTrue(all(r['split']=='development' for r in split if r['id'] in KNOWN_DEVELOPMENT))


if __name__=='__main__':
    unittest.main()
