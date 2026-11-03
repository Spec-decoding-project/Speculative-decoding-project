"""Conventional marginal-probability tree reference, preserving q0's chain.

Uses only original-drafter top-k tokens and full-vocabulary log probabilities.
Not a new method, a calibrated acceptance model, or an implemented GPU decoder.
"""
import heapq
import math


def build(original, candidate_ids, log_probabilities, *, node_budget=30, fixed_x1=False):
    length=len(original)
    if length<3 or len(candidate_ids)!=length-1 or len(log_probabilities)!=length-1:
        raise ValueError('One row for every future slot')
    if node_budget<length:raise ValueError('Budget cannot preserve original chain')
    for i,(ids,scores) in enumerate(zip(candidate_ids,log_probabilities),1):
        if not ids or len(ids)!=len(scores) or len(set(ids))!=len(ids) or ids[0]!=original[i]:
            raise ValueError('Unique candidates must begin with original argmax')
        if any(not math.isfinite(s) or s>1e-7 for s in scores) or any(scores[j]<scores[j+1] for j in range(len(scores)-1)):
            raise ValueError('Sorted finite log probabilities required')
        if sum(math.exp(s) for s in scores)>1.00001:raise ValueError('Use full-vocabulary probabilities, not unnormalized scores')
    nodes={tuple(original[:i]) for i in range(1,length+1)}
    mass={tuple(original[:1]):0.}
    for depth in range(1,length):mass[tuple(original[:depth+1])]=mass[tuple(original[:depth])]+log_probabilities[depth-1][0]
    heap=[]
    def offer(parent):
        depth=len(parent)
        if depth==length or (fixed_x1 and depth==1):return
        for token,score in zip(candidate_ids[depth-1],log_probabilities[depth-1]):
            child=parent+(token,)
            if child not in nodes:heapq.heappush(heap,(-(mass[parent]+score),child))
    for parent in sorted(nodes):offer(parent)
    while heap and len(nodes)<node_budget:
        negative,path=heapq.heappop(heap)
        if path in nodes:continue
        if path[:-1] not in nodes:raise AssertionError('Missing ancestral context')
        nodes.add(path);mass[path]=-negative;offer(path)
    return dict(nodes=sorted(nodes,key=lambda p:(len(p),p)),log_mass=mass,
                node_count=len(nodes),scope='Independent positional probability ranking; no target labels or new model pass')


def matching_prefix(nodes, labels, valid):
    if not valid[0] or valid!=[True]*sum(valid)+[False]*(len(valid)-sum(valid)):raise ValueError('Reference mask')
    nodes={tuple(p) for p in nodes};count=0
    for i in range(1,sum(valid)+1):
        if tuple(labels[:i]) not in nodes:break
        count=i
    return count
