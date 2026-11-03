"""Fixed-address target verifier using upstream StaticCache and CUDA graphs.

No altered weights, sampling rule, precision, or attention semantics. Padding,
input copies and in-place cache gathering are explicit execution choices.
"""
import time


class StaticVerifier:
    def __init__(self,target,prefix_state,nodes=128,capacity=4096):
        import torch
        from transformers import StaticCache
        self.target=target;self.nodes=nodes;self.capacity=capacity
        self.cache=StaticCache(config=target.config,max_cache_len=capacity)
        for layer,(keys,values) in zip(self.cache.layers,prefix_state,strict=True):
            layer.lazy_initialization(keys,values)
        self.ids=torch.empty((1,nodes),device=target.device,dtype=torch.long)
        self.positions=torch.empty_like(self.ids)
        self.mask=torch.full((1,1,nodes,capacity),torch.finfo(target.dtype).min,device=target.device,dtype=target.dtype)
        self.graph=None;self.output=None;self.setup_seconds=None
        self.load_prefix(prefix_state)

    def addresses(self):
        return [(layer.keys.data_ptr(),layer.values.data_ptr(),layer.cumulative_length.data_ptr()) for layer in self.cache.layers]

    def load_prefix(self,state):
        past=state[0][0].shape[-2]
        if past+self.nodes>self.capacity:raise ValueError('Static verifier capacity exceeded')
        for layer,(keys,values) in zip(self.cache.layers,state,strict=True):
            layer.keys[...,:past,:].copy_(keys)
            layer.values[...,:past,:].copy_(values)
            layer.cumulative_length.fill_(past)
        self.past=past

    def copy_inputs(self,ids,positions,attention):
        import torch
        if tuple(ids.shape)!=(1,self.nodes) or tuple(positions.shape)!=tuple(ids.shape):raise ValueError('Fixed query shape')
        if tuple(attention.shape)!=(1,1,self.nodes,self.past+self.nodes):raise ValueError('Logical attention shape')
        self.ids.copy_(ids);self.positions.copy_(positions)
        self.mask.fill_(torch.finfo(self.target.dtype).min)
        self.mask[...,:self.past+self.nodes].copy_(attention)

    def _forward(self):
        return self.target(self.ids,position_ids=self.positions,attention_mask=self.mask,
            past_key_values=self.cache,use_cache=True,output_hidden_states=True)

    def prepare_graph(self,prefix_state,ids,positions,attention):
        import torch
        if self.target.device.type!='cuda':raise ValueError('CUDA capture requires CUDA; no CPU surrogate')
        if self.graph is not None:raise ValueError('Capture exactly once')
        torch.cuda.synchronize();started=time.perf_counter()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.load_prefix(prefix_state);self.copy_inputs(ids,positions,attention);self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.load_prefix(prefix_state);self.copy_inputs(ids,positions,attention)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=self._forward()
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-started
        self.load_prefix(prefix_state)
        return dict(seconds=self.setup_seconds,warmup_forwards=3,capture_forwards=1,nodes=self.nodes,capacity=self.capacity)

    def __call__(self,ids,positions,attention):
        self.copy_inputs(ids,positions,attention)
        if self.graph is None:return self._forward()
        self.graph.replay()
        return self.output

    def compact(self,past,nodes):
        import torch
        # Preserve pointers used by the captured graph. The index_select RHS is
        # materialized before copying, so overlap cannot corrupt later source rows.
        indices=torch.tensor(nodes,device=self.target.device,dtype=torch.long)+past
        for layer in self.cache.layers:
            layer.keys[...,past:past+len(nodes),:].copy_(layer.keys.index_select(2,indices))
            layer.values[...,past:past+len(nodes),:].copy_(layer.values.index_select(2,indices))
            layer.cumulative_length.fill_(past+len(nodes))
        self.past=past+len(nodes)
