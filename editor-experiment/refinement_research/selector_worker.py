"""Pinned batch-one vLLM adapter: decide before any editor context copy or step."""
import functools
import time

from refinement_research.native_speed_worker import NativeSpeedWorker
from refinement_research.selector import compile_first_only, features, predict, selected


class SelectorWorker(NativeSpeedWorker):
    def init_worker_extension(self):
        super().init_worker_extension()
        import torch
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
        worker=self
        self.selector_model=None
        self.selector_gaps=None
        original=DraftModelSpeculator.compute_draft_logits

        @functools.wraps(original)
        def compute(speculator, *args, **kwargs):
            logits=original(speculator,*args,**kwargs)
            if worker.measuring and worker.mode in ('shadow','selected','selector_probe'):
                if logits.ndim != 2 or logits.shape != (15,151936):
                    raise RuntimeError('Unexpected DFlash vocabulary scores')
                values=torch.topk(logits,k=2,dim=-1,sorted=True).values.float()
                worker.selector_gaps=values[:,0]-values[:,1]
            return logits
        DraftModelSpeculator.compute_draft_logits=compute

    def selector_load(self, model):
        from copy_action_fit.data import digest
        from refinement_research.selector import FEATURES
        if model['features'] != FEATURES or model['schema'] != 'cheap-selector-v1':
            raise ValueError('Incompatible selector artifact')
        self.selector_model=model
        self.compiled_selector=compile_first_only(model)
        return dict(selector_digest=digest(model),compiled=self.compiled_selector)

    def selector_request(self, prompt_length):
        self.selector_prompt_length=prompt_length

    def speed_begin(self, *args, **kwargs):
        self.selector_decisions=[]
        self.selector_gaps=None
        return super().speed_begin(*args,**kwargs)

    def choose_edit(self, speculator, result, batch, args):
        if self.mode not in ('shadow','selected','selector_probe'):
            return True
        started=time.perf_counter()
        if self.selector_gaps is None or speculator.use_local_argmax_reduction:
            raise RuntimeError('DFlash confidence features were not computed')
        states=self.model_runner.req_states
        slot=states.req_id_to_index[batch.req_ids[0]]
        length=int(states.total_len.gpu[slot])
        proposal=result[0].tolist()
        gaps=self.selector_gaps.tolist()
        self.selector_gaps=None
        x=features(proposal,gaps,int(args[5][0]),self.selector_prompt_length,length)
        score=None if self.selector_model is None else predict(self.selector_model['tree'],x)
        decision=(self.mode=='shadow' or (self.mode=='selected' and selected(self.selector_model,x)))
        self.selector_decisions.append(dict(length=length,features=x,score=score,selected=decision,
            host_ms_including_pending_native_wait=1000*(time.perf_counter()-started)))
        return decision

    def speed_finish(self):
        result=super().speed_finish()
        result['selector_decisions']=self.selector_decisions
        return result
