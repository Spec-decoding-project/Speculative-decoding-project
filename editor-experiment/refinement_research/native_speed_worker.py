"""Bounded batch-one DFlash replacement experiment for pinned vLLM 0.34.

Only candidate IDs are changed, after DFlash proposes and before native
verification. Target/draft caches and the rejection sampler remain native.
"""
from collections import OrderedDict
import functools
import time

from full_corpus_fit.capture import CorpusCaptureWorker
from refinement_research.editor_kernels import editor_kernels


def editor_input(committed, proposal):
    if len(committed) < 2 or len(proposal) != 15:
        raise ValueError('Require full committed context and 15 native proposals')
    # The final committed token is the target-selected anchor, not editor context.
    return committed[max(0, len(committed)-1-1024):-1], [committed[-1], *proposal]


class NativeSpeedWorker(CorpusCaptureWorker):
    def init_worker_extension(self):
        super().init_worker_extension()
        import torch
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
        worker = self
        self.mode = 'off'
        self.measuring = False
        self.editor = None
        self.graphs = OrderedDict()
        self.precision = None
        self.kernels = 'invariant'
        original_propose = DFlashSpeculator.propose
        original_execute = GPUModelRunner.execute_model
        original_sample_tokens = GPUModelRunner.sample_tokens
        original_sample = GPUModelRunner.sample

        @functools.wraps(original_propose)
        def propose(speculator, *args, **kwargs):
            if worker.measuring and worker.details and worker.decode_started is None:
                # A single synchronized boundary after the final prefill's target
                # token, before DFlash and editor work. Same probe in both arms.
                torch.cuda.synchronize()
                if int(args[5][0]) > 0:
                    worker.decode_started = time.perf_counter()
            result = original_propose(speculator, *args, **kwargs)
            if not worker.measuring or worker.mode == 'off' or kwargs.get('dummy_run', False):
                return result
            batch = args[0]
            if batch.num_reqs != 1 or result.shape != (1, 15):
                raise RuntimeError('Native editor pilot only supports batch one / b16')
            if kwargs.get('num_speculative_tokens', 15) == 0:
                return result
            started = time.perf_counter()
            # num_sampled is zero for an unfinished chunked prefill.
            if int(args[5][0]) == 0:
                return result
            states = worker.model_runner.req_states
            slot = states.req_id_to_index[batch.req_ids[0]]
            length = int(states.total_len.gpu[slot])
            committed = states.all_token_ids.gpu[slot, :length].cpu().tolist()
            prefix, draft = editor_input(committed, result[0].tolist())
            if worker.reference is not None:
                # The last native round can compute past the request's output cap.
                known = min(length, len(worker.reference))
                if committed[:known] != worker.reference[:known]:
                    raise RuntimeError('Native committed state differs from independent greedy prefix')
            cold_ms = 0.0
            if worker.mode == 'identity':
                edited = draft
            elif worker.mode in ('oracle', 'reject_first'):
                continuation = worker.reference[length:length+15]
                edited = [draft[0], *continuation, *draft[1+len(continuation):]]
                if worker.mode == 'reject_first':
                    edited[1] = (edited[1]+1) % 151936
            elif worker.mode == 'editor':
                from refinement_research.graph_replay import CapturedEditor
                key = len(prefix)
                if key not in worker.graphs:
                    worker.misses += 1
                    if len(worker.graphs) == 64:
                        worker.graphs.popitem(last=False)
                        worker.evictions += 1
                    before = time.perf_counter()
                    with editor_kernels(worker.kernels):
                        worker.graphs[key] = CapturedEditor(worker.editor, prefix, draft)
                    cold_ms = 1000*(time.perf_counter()-before)
                else:
                    worker.hits += 1
                    worker.graphs.move_to_end(key)
                edited = worker.graphs[key].propose(prefix, draft)
                if worker.active:
                    with editor_kernels(worker.kernels):
                        eager = worker.editor.propose(prefix, draft)
                    if eager != edited:
                        raise RuntimeError('Native graph output differs from eager editor')
                if edited[:2] != draft[:2]:
                    raise RuntimeError('Editor changed anchor/first proposal')
            else:
                raise ValueError('Unknown native mode')
            # Return a fresh tensor: never mutate DFlash's persistent output buffer.
            result = torch.tensor([edited[1:]], dtype=result.dtype, device=result.device)
            worker.edit_calls += 1
            worker.editor_ms.append(1000*(time.perf_counter()-started))
            worker.cold_ms += cold_ms
            if worker.active:
                worker.edits.append(dict(anchor=length-1, prefix=committed[:-1],
                    original=draft, edited=edited, graph_preparation_ms=cold_ms))
            return result

        @functools.wraps(original_execute)
        def execute(runner, *args, **kwargs):
            if worker.measuring:
                from vllm.model_executor.determinism import batch_invariant
                if not batch_invariant._batch_invariant_MODE or batch_invariant._batch_invariant_LIB is None:
                    raise RuntimeError('Target execution lost invariant dispatch')
                worker.round_started = time.perf_counter()
                worker.round_event = torch.cuda.Event(enable_timing=True)
                worker.round_event.record()
                worker.round_speculative = False
            return original_execute(runner, *args, **kwargs)

        @functools.wraps(original_sample)
        def sample(runner, hidden_states, batch, grammar_output):
            result = original_sample(runner, hidden_states, batch, grammar_output)
            if worker.measuring:
                worker.round_speculative = batch.num_draft_tokens > 0
                if worker.details:
                    worker.all_samples.append((worker.round_speculative, result[1].clone()))
                if worker.round_speculative:
                    # Copies stay on device until finish; no per-round CPU sync.
                    worker.acceptance.append((result[1].clone(), result[2].clone()))
            return result

        @functools.wraps(original_sample_tokens)
        def sample_tokens(runner, *args, **kwargs):
            result = original_sample_tokens(runner, *args, **kwargs)
            if worker.measuring:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                worker.rounds.append((worker.round_event, end,
                    1000*(time.perf_counter()-worker.round_started), worker.round_speculative))
            return result

        DFlashSpeculator.propose = propose
        GPUModelRunner.execute_model = execute
        GPUModelRunner.sample = sample
        GPUModelRunner.sample_tokens = sample_tokens

    def speed_load(self, initial, gate, precision, kernels='invariant',
                   expected_gate_sha256='b6f1c27a312a829efedc1db2ac42091f3702ad7d586f9025033508ab2c606882'):
        import torch
        from copy_action_fit.data import CHECKPOINT_SHA256, sha
        from refinement_research.train_gates import initialize
        if precision not in ('fp32', 'bf16') or sha(initial) != CHECKPOINT_SHA256:
            raise ValueError('Invalid editor source or precision')
        self.graphs.clear()
        self.editor = None
        torch.cuda.empty_cache()
        self.editor = initialize(initial, 'linear')
        payload = torch.load(gate, map_location='cpu', weights_only=True)
        if sha(gate) != expected_gate_sha256:
            raise ValueError('Gate checkpoint changed')
        self.editor.gate_head.load_state_dict(payload['gate'], strict=True)
        if precision == 'bf16':
            self.editor.to(dtype=torch.bfloat16)
            self.editor.gate_head.float()
        self.editor.requires_grad_(False)
        self.precision = precision
        self.kernels = kernels
        return dict(precision=precision, initial_sha256=sha(initial), gate_sha256=sha(gate),
                    kernels=kernels, generator_dtype=str(self.editor.token.weight.dtype),
                    parameters=sum(p.numel() for p in self.editor.parameters()))

    def speed_begin(self, mode, audit=False, reference=None, clear_graphs=False, details=False):
        import torch
        if clear_graphs:
            self.graphs.clear()
            torch.cuda.empty_cache()
        self.mode, self.reference = mode, reference
        self.details = details
        self.decode_started = None
        self.all_samples = []
        self.rounds, self.edits, self.editor_ms, self.acceptance = [], [], [], []
        self.hits = self.misses = self.evictions = self.edit_calls = 0
        self.cold_ms = 0.0
        if audit:
            self.corpus_begin(self.kernels == 'standard')
        else:
            self.active = False
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.measuring = True
        return dict(graphs=len(self.graphs), precision=self.precision)

    def speed_stop(self):
        # Drain this worker's GPU context, including any last-round proposal.
        # The timestamp excludes serializing the much larger diagnostic payload.
        import torch
        torch.cuda.synchronize()
        stamp = time.perf_counter()
        self.measuring = False
        return stamp

    def speed_finish(self):
        import torch
        torch.cuda.synchronize()
        self.measuring = False
        audited = self.corpus_finish() if self.active else None
        return dict(mode=self.mode, precision=self.precision, kernels=self.kernels,
            decode_started=self.decode_started,
            all_samples=[dict(speculative=spec, sampled=int(n.sum())) for spec,n in self.all_samples],
            rounds=[dict(gpu_ms=a.elapsed_time(b), host_ms=host, speculative=spec)
                    for a,b,host,spec in self.rounds],
            sampled=[int(a.sum()) for a,b in self.acceptance],
            rejected=[int(b.sum()) for a,b in self.acceptance],
            edits=self.edits, editor_ms=self.editor_ms, graph_preparation_ms=self.cold_ms,
            graph_hits=self.hits, graph_misses=self.misses, graph_evictions=self.evictions,
            graph_count=len(self.graphs), edit_calls=self.edit_calls,
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), native=audited)
