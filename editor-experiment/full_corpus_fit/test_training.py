from pathlib import Path
import errno
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import torch

from autoregressive_editor.learning import Trainer
from autoregressive_editor.model import Config
from full_corpus_fit.train import epoch_order, learning_rate, validate_cursor


class ResumeContract(unittest.TestCase):
    def test_storage_failure_preserves_last_resumable_checkpoint(self):
        """A partial new save must never replace the last committed weights."""
        torch.set_num_threads(1)
        row=dict(prefix=[1],draft=[4,5,6,7],labels=[4,5,8,9])
        trainer=Trainer(Config(vocab_size=16,width=8,layers=1,heads=2,max_length=20),
                        'draft',device='cpu')
        trainer.update([row])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            trainer.save(path,dict(epoch=0,cursor=32))
            original=hashlib.sha256(path.read_bytes()).hexdigest()
            trainer.update([row])

            def exhausted_save(payload, temporary):
                Path(temporary).write_bytes(b'interrupted checkpoint')
                raise OSError(errno.ENOSPC, 'No space left on device')

            with patch('autoregressive_editor.learning.torch.save',side_effect=exhausted_save):
                with self.assertRaises(OSError):
                    trainer.save(path,dict(epoch=0,cursor=64))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original)
            restored,state=Trainer.load(path,device='cpu')
            self.assertEqual(restored.step,1)
            self.assertEqual(state,dict(epoch=0,cursor=32))
            # Recovery overwrites the partial temporary file and becomes loadable.
            trainer.save(path,dict(epoch=0,cursor=64))
            recovered,state=Trainer.load(path,device='cpu')
            self.assertEqual(recovered.step,2)
            self.assertEqual(state,dict(epoch=0,cursor=64))
            self.assertTrue(all(torch.equal(value,recovered.model.state_dict()[name])
                                for name,value in trainer.model.state_dict().items()))

    def test_restored_mid_epoch_run_matches_uninterrupted_weights(self):
        torch.set_num_threads(1)
        rows=[dict(prefix=[1+i%3],draft=[4,5,6,7],labels=[4,5,8+i%3,9]) for i in range(70)]
        config=Config(vocab_size=16,width=8,layers=1,heads=2,max_length=20)
        order=epoch_order(list(range(70)),0)
        left=Trainer(config,'draft',device='cpu')
        for group in left.optimizer.param_groups:
            group['lr']=learning_rate(1,9)
            group['weight_decay']=.01
        left.update([rows[i] for i in order[:32]])
        state=dict(epoch=0,cursor=32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            left.save(path,state)
            restored,saved=Trainer.load(path,device='cpu')
        validate_cursor(saved,restored.step,len(rows))
        for trainer in (left,restored):
            for cursor in (32,64):
                for group in trainer.optimizer.param_groups:
                    group['lr']=learning_rate(trainer.step+1,9)
                trainer.update([rows[i] for i in order[cursor:cursor+32]])
        self.assertEqual(left.step,3)
        self.assertTrue(all(torch.equal(value,restored.model.state_dict()[name])
                            for name,value in left.model.state_dict().items()))

    def test_invalid_cursor_cannot_skip_examples(self):
        validate_cursor(dict(epoch=0,cursor=64),2,70)
        validate_cursor(dict(epoch=0,cursor=70),3,70)
        validate_cursor(dict(epoch=1,cursor=0),3,70)
        for state,step in [(dict(epoch=0,cursor=31),1),(dict(epoch=1,cursor=0),2),
                           (dict(epoch=3,cursor=32),10)]:
            with self.assertRaises(ValueError):
                validate_cursor(state,step,70)

    def test_epoch_shuffle_and_schedule_are_reproducible(self):
        ids=list(range(70))
        self.assertEqual(epoch_order(ids,1),epoch_order(ids,1))
        self.assertNotEqual(epoch_order(ids,0),epoch_order(ids,1))
        self.assertEqual(sorted(epoch_order(ids,1)),ids)
        self.assertLess(learning_rate(1,1000),learning_rate(20,1000))
        self.assertAlmostEqual(learning_rate(1000,1000),.0001)


if __name__=='__main__':
    unittest.main()
