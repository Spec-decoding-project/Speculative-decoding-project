"""One substitution maximizing a product-marginal added-prefix proxy."""
import math

def choose(original,candidate_ids,log_probabilities):
    if len(candidate_ids)!=len(original)-1 or len(log_probabilities)!=len(candidate_ids):
        raise ValueError('One row per drafted future position')
    for j,(ids,logs) in enumerate(zip(candidate_ids,log_probabilities),1):
        if len(ids)!=4 or len(logs)!=4 or ids[0]!=original[j] or len(set(ids))!=4:
            raise ValueError('Four unique original candidates required')
        if any(not math.isfinite(x) or x>1e-6 for x in logs):raise ValueError('Finite full-vocabulary log probabilities required')
    best=None
    for j in range(2,len(original)):
        prefix=sum(log_probabilities[i-1][0] for i in range(1,j))
        tail=1.;running=0.
        for end in range(j+1,len(original)):
            running+=log_probabilities[end-1][0];tail+=math.exp(running)
        for rank in (1,2,3):
            score=prefix+log_probabilities[j-1][rank]+math.log(tail)
            candidate=(score,-j,-rank)
            if best is None or candidate>best:best=candidate
    score,negative_j,negative_rank=best;j=-negative_j;rank=-negative_rank
    proposal=list(original);proposal[j]=candidate_ids[j-1][rank]
    return dict(position=j,rank=rank+1,token=proposal[j],proposal=proposal,log_proxy_utility=score)
