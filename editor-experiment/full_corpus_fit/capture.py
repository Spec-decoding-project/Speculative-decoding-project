"""Observe native vLLM requests without modifying logits, candidates or sampling.

Uses the same pinned GPUModelRunner.sample hook as the completed small fits.
Full logits are hashed only in the calibration/reference check. Corpus capture
stores raw argmax/prefix evidence and actual candidate inputs in bounded chunks.
"""
import functools
import hashlib
import json
import time


class CorpusCaptureWorker:
    def init_worker_extension(self):
        import torch
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        worker = self
        self.active = False
        self.full_hashes = False
        self.context = None
        self.histories = {}
        self.trace = []
        self.blocks = []
        self.calls = 0
        self.started = time.monotonic()
        original_forward = Qwen3ForCausalLM.forward
        original_logits = Qwen3ForCausalLM.compute_logits
        original_sample = GPUModelRunner.sample

        @functools.wraps(original_forward)
        def forward(model, *args, **kwargs):
            worker.calls += 1
            return original_forward(model, *args, **kwargs)

        @functools.wraps(original_logits)
        def logits(model, *args, **kwargs):
            result = original_logits(model, *args, **kwargs)
            if worker.active and worker.context is not None:
                if result is None or result.shape != (len(worker.context), 151936):
                    raise RuntimeError('Unrecognized target logit layout')
                choices = result.argmax(-1).cpu().tolist()
                raw = result.detach().cpu().contiguous() if worker.full_hashes else None
                for i, (metadata, token) in enumerate(zip(worker.context, choices)):
                    row = dict(**metadata, argmax=token, vocabulary_size=151936)
                    if raw is not None:
                        row['logits_sha256'] = hashlib.sha256(raw[i].view(torch.uint8).numpy().tobytes()).hexdigest()
                    worker.trace.append(row)
            return result

        @functools.wraps(original_sample)
        def sampled(runner, hidden_states, input_batch, grammar_output):
            blocks, metadata = [], []
            if worker.active:
                starts = input_batch.query_start_loc.cpu().tolist()
                ids = input_batch.input_ids.cpu().tolist()
                positions = input_batch.positions.cpu().tolist()
                selected = input_batch.logits_indices.cpu().tolist()
                wanted, by_index = set(selected), {}
                for i, request_id in enumerate(input_batch.req_ids):
                    history = worker.histories.setdefault(request_id, [])
                    for j in range(starts[i], starts[i+1]):
                        position = positions[j]
                        if position > len(history):
                            raise RuntimeError('Gap in reconstructed native context')
                        del history[position:]
                        history.append(ids[j])
                        if j in wanted:
                            by_index[j] = dict(request_id=request_id, position=position,
                                prefix_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest())
                metadata = [by_index[j] for j in selected]
                counts = input_batch.num_draft_tokens_per_req
                if counts is not None:
                    for i, request_id in enumerate(input_batch.req_ids):
                        if int(counts[i]) != 15:
                            continue
                        lo, hi = starts[i], starts[i+1]
                        if hi-lo != 16 or positions[lo:hi] != list(range(positions[lo], positions[lo]+16)):
                            raise RuntimeError('Unexpected full native block layout')
                        blocks.append(dict(request_id=request_id, batch_index=i, anchor=positions[lo],
                            prefix=list(worker.histories[request_id][:positions[lo]]), draft=ids[lo:hi],
                            positions=positions[lo:hi], native_draft_count=15))
            worker.context = metadata if worker.active else None
            try:
                result = original_sample(runner, hidden_states, input_batch, grammar_output)
                if worker.active:
                    accepted, rejected = result[1].cpu().tolist(), result[2].cpu().tolist()
                    for block in blocks:
                        index = block['batch_index']
                        block.update(num_sampled=accepted[index], num_rejected=rejected[index])
                        worker.blocks.append(block)
                return result
            finally:
                worker.context = None

        Qwen3ForCausalLM.forward = forward
        Qwen3ForCausalLM.compute_logits = logits
        GPUModelRunner.sample = sampled

    def corpus_begin(self, full_hashes=False):
        self.active, self.full_hashes = True, full_hashes
        self.histories, self.trace, self.blocks = {}, [], []
        return dict(calls=self.calls)

    def corpus_finish(self):
        import torch
        from vllm.model_executor.determinism import batch_invariant
        model = self.model_runner.model
        attention = []
        for name, module in model.named_modules():
            impl = getattr(module, 'impl', None)
            if impl is not None and 'attention' in type(impl).__module__:
                attention.append(dict(name=name, cls=type(impl).__name__,
                    batch_invariant=getattr(impl, 'batch_invariant_enabled', None)))
        self.active = False
        return dict(calls=self.calls, trace=self.trace, blocks=self.blocks,
            model_class=type(model).__name__, invariant_active=batch_invariant._batch_invariant_MODE,
            attention=attention, peak_allocated_bytes=torch.cuda.max_memory_allocated())
