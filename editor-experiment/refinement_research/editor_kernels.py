"""Experimental scoped editor dispatch for the pinned, synchronous worker.

vLLM 0.34 installs process-wide ATen overrides. A graph records its kernels at
capture, so capture the editor under ordinary ATen dispatch, then restore the
invariant target dispatch before any native model execution. This uses private
pinned APIs and is intentionally restricted to this single-threaded pilot.
"""
from contextlib import contextmanager


@contextmanager
def editor_kernels(mode, invariant=None):
    if mode == 'invariant':
        yield
        return
    if mode != 'standard':
        raise ValueError('Unknown editor kernel mode')
    if invariant is None:
        from vllm.model_executor.determinism import batch_invariant as invariant
    if not invariant._batch_invariant_MODE or invariant._batch_invariant_LIB is None:
        raise RuntimeError('Expected active invariant target dispatch')
    invariant._batch_invariant_LIB._destroy()
    invariant._batch_invariant_LIB = None
    invariant._batch_invariant_MODE = False
    try:
        yield
    finally:
        invariant.enable_batch_invariant_mode()
        if not invariant._batch_invariant_MODE or invariant._batch_invariant_LIB is None:
            raise RuntimeError('Failed to restore invariant target dispatch')
