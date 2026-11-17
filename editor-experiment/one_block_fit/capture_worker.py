"""Native candidate capture for the authorized one-block fit.

Derived from the executed vllm_contract_worker.py; the old file stays immutable.

Installed before model loading, so profile/warmup target forwards count too.
No logits, proposals, cache entries, or sampling decisions are changed.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class FitCaptureWorker:
    def init_worker_extension(self):
        import torch
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        path = Path(os.environ['VLLM_PROBE_BUDGET'])
        state = json.loads(path.read_text())
        if state['max_calls'] != 500 or state['max_seconds'] != 120:
            raise RuntimeError('Unreviewed one-block collection budget')
        self.probe_state = state
        self.probe_phase = 'startup'
        self.probe_capture = False
        self.probe_trace = []
        self.probe_context = None
        self.probe_histories = {}
        self.probe_timings = []
        self.blocks = []
        worker = self

        def wrap(method, is_forward):
            @functools.wraps(method)
            def measured(model, *args, **kwargs):
                if type(model) is not Qwen3ForCausalLM:
                    raise RuntimeError('Unexpected target class: budget coverage unverified')
                if state['calls'] >= state['max_calls'] or state['seconds'] >= state['max_seconds']:
                    raise RuntimeError('Diagnostic target budget exhausted')
                torch.cuda.synchronize()
                started = time.monotonic()
                if is_forward:
                    state['calls'] += 1
                    save(path, state)
                try:
                    result = method(model, *args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - started
                    state['seconds'] += seconds
                    worker.probe_timings.append(dict(phase=worker.probe_phase,
                        kind='forward' if is_forward else 'head', seconds=seconds))
                    save(path, state)
                if not is_forward and worker.probe_capture and worker.probe_context is not None:
                    rows = worker.probe_context
                    if result is None or len(rows) != result.shape[0]:
                        raise RuntimeError('Cannot align raw logit rows')
                    raw = result.detach().cpu().contiguous()
                    for metadata, row in zip(rows, raw):
                        top = row.float().topk(5)
                        worker.probe_trace.append(dict(**metadata, dtype=str(row.dtype),
                            vocabulary_size=row.numel(),
                            logits_sha256=hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest(),
                            argmax=int(row.argmax()),
                            top5=list(zip(top.indices.tolist(), top.values.tolist()))))
                return result
            return measured

        Qwen3ForCausalLM.forward = wrap(Qwen3ForCausalLM.forward, True)
        Qwen3ForCausalLM.compute_logits = wrap(Qwen3ForCausalLM.compute_logits, False)
        original_sample = GPUModelRunner.sample

        @functools.wraps(original_sample)
        def sampled(runner, hidden_states, input_batch, grammar_output):
            metadata, blocks = [], []
            if worker.probe_capture:
                starts = input_batch.query_start_loc.cpu().tolist()
                ids = input_batch.input_ids.cpu().tolist()
                positions = input_batch.positions.cpu().tolist()
                by_index = {}
                for i, request_id in enumerate(input_batch.req_ids):
                    history = worker.probe_histories.setdefault(request_id, [])
                    for j in range(starts[i], starts[i+1]):
                        position = positions[j]
                        if position > len(history):
                            raise RuntimeError('Gap in reconstructed target prefix')
                        del history[position:]
                        history.append(ids[j])
                        by_index[j] = dict(request_id=request_id, position=position,
                            prefix_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest())
                metadata = [by_index[j] for j in input_batch.logits_indices.cpu().tolist()]
                counts = input_batch.num_draft_tokens_per_req
                if counts is not None:
                    for i, request_id in enumerate(input_batch.req_ids):
                        if int(counts[i]) != 15:
                            continue
                        lo, hi = starts[i], starts[i+1]
                        if hi-lo != 16 or positions[lo:hi] != list(range(positions[lo],positions[lo]+16)):
                            raise RuntimeError('Unexpected full native verification layout')
                        blocks.append(dict(request_id=request_id, batch_index=i,
                            anchor=positions[lo], prefix=list(worker.probe_histories[request_id][:positions[lo]]),
                            draft=ids[lo:hi], positions=positions[lo:hi],
                            native_draft_count=15, phase=worker.probe_phase))
            worker.probe_context = metadata if worker.probe_capture else None
            try:
                result = original_sample(runner, hidden_states, input_batch, grammar_output)
                if worker.probe_capture:
                    sampled_counts = result[1].cpu().tolist()
                    rejected_counts = result[2].cpu().tolist()
                    for block in blocks:
                        i = block['batch_index']
                        block.update(num_sampled=sampled_counts[i], num_rejected=rejected_counts[i])
                        worker.blocks.append(block)
                    worker.probe_trace.append(dict(sampling=True,
                        request_ids=list(input_batch.req_ids),
                        num_sampled=result[1].cpu().tolist(),
                        num_rejected=result[2].cpu().tolist()))
                return result
            finally:
                worker.probe_context = None
        GPUModelRunner.sample = sampled

    def probe_begin(self, phase, capture):
        self.probe_phase = phase
        self.probe_capture = capture
        self.probe_histories = {}
        self.probe_trace = []
        self.probe_timings = []
        self.blocks = []
        return dict(self.probe_state)

    def probe_finish(self):
        import torch
        from vllm.model_executor.determinism import batch_invariant
        model = self.model_runner.model
        implementations = []
        for name, module in model.named_modules():
            impl = getattr(module, 'impl', None)
            if impl is not None and 'attention' in type(impl).__module__:
                implementations.append(dict(name=name, cls=type(impl).__name__,
                    batch_invariant=getattr(impl, 'batch_invariant_enabled', None),
                    flash_attention_version=getattr(impl, 'vllm_flash_attn_version', None)))
        return dict(budget=dict(self.probe_state), trace=self.probe_trace, blocks=self.blocks,
            timings=self.probe_timings, model_class=type(model).__name__,
            invariant_active=batch_invariant._batch_invariant_MODE,
            attention=implementations,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
            reduced_bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
