"""Fixed two-chain greedy verification layout for conditional redrafting.

This is model/provider independent. It does not launch jobs or import torch.
The target prefix cache excludes the known anchor. The root query is the
anchor, and logits at a node predict its CHILD token, never its own token.
Two suffixes occupy separate physical rows even when token values coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _token_tuple(tokens: Sequence[int]) -> tuple[int, ...]:
    result = tuple(tokens)
    if any(type(t) is not int or t < 0 for t in result):
        raise ValueError("Token IDs must be nonnegative Python integers")
    return result


@dataclass(frozen=True)
class GreedyDecision:
    branch: str
    original_accepted: int
    refined_accepted: int
    accepted_nodes: tuple[int, ...]  # Includes anchor; excludes bonus token.
    accepted_tokens: tuple[int, ...]
    bonus_token: int | None

    @property
    def emitted_tokens(self) -> tuple[int, ...]:
        """New committed sequence, including the previously pending anchor."""
        if self.bonus_token is None:
            return self.accepted_tokens
        return self.accepted_tokens + (self.bonus_token,)


@dataclass(frozen=True)
class TwoChainLayout:
    horizon: int
    cut: int
    parents: tuple[int, ...]
    depths: tuple[int, ...]
    original_nodes: tuple[int, ...]
    refined_nodes: tuple[int, ...]

    @classmethod
    def build(cls, horizon: int, cut: int = 1) -> TwoChainLayout:
        if type(horizon) is not int or type(cut) is not int:
            raise ValueError("Horizon and cut must be integers")
        if not 1 <= cut < horizon:
            raise ValueError("Require 1 <= cut < horizon; horizon excludes anchor")
        parents = [-1] + list(range(horizon))
        depths = list(range(horizon + 1))
        original = tuple(range(horizon + 1))
        refined = list(range(cut + 1))
        for depth in range(cut + 1, horizon + 1):
            parents.append(refined[-1])
            depths.append(depth)
            refined.append(len(parents) - 1)
        return cls(horizon, cut, tuple(parents), tuple(depths), original, tuple(refined))

    @property
    def node_count(self) -> int:
        return len(self.parents)

    def pack_tokens(
        self, anchor: int, original: Sequence[int], refined: Sequence[int]
    ) -> tuple[int, ...]:
        anchor_id, = _token_tuple((anchor,))
        original = _token_tuple(original)
        refined = _token_tuple(refined)
        if len(original) != self.horizon or len(refined) != self.horizon:
            raise ValueError("Each chain must contain exactly horizon future tokens")
        if original[:self.cut] != refined[:self.cut]:
            raise ValueError("Refinement changed the clamped original prefix")
        return (anchor_id,) + original + refined[self.cut:]

    def positions(self, past_length: int) -> tuple[int, ...]:
        if type(past_length) is not int or past_length < 0:
            raise ValueError("Invalid prefix cache length")
        return tuple(past_length + d for d in self.depths)

    def visible_nodes(self, node: int) -> tuple[int, ...]:
        if type(node) is not int or not 0 <= node < self.node_count:
            raise ValueError("Invalid node index")
        path = []
        while node != -1:
            path.append(node)
            node = self.parents[node]
        return tuple(reversed(path))

    def allowed_attention(self, past_length: int) -> tuple[tuple[bool, ...], ...]:
        """True means visible, including all committed context and the node itself.

        A tensor adapter must turn False into the backend's masked value. Logical
        positions are NOT physical cache slots after the paths branch.
        """
        self.positions(past_length)  # Validate independently of mask allocation.
        prefix = (True,) * past_length
        rows = []
        for node in range(self.node_count):
            ancestors = set(self.visible_nodes(node))
            rows.append(prefix + tuple(i in ancestors for i in range(self.node_count)))
        return tuple(rows)

    def decide(
        self,
        packed_tokens: Sequence[int],
        next_token_by_node: Sequence[int],
        *,
        eos_token_ids: Sequence[int] = (),
        max_new_tokens: int | None = None,
    ) -> GreedyDecision:
        """Choose the longest verified chain; retain original on length ties.

        Predictions must come from the ACTUAL target scores for these masked
        nodes. This is greedy verification, not a stochastic rejection sampler.
        Apply output limits BEFORE constructing the cache-compaction selection.
        """
        tokens = _token_tuple(packed_tokens)
        predictions = _token_tuple(next_token_by_node)
        eos = frozenset(_token_tuple(eos_token_ids))
        if len(tokens) != self.node_count or len(predictions) != self.node_count:
            raise ValueError("Need a token and next-token prediction for every node")
        if max_new_tokens is not None and (
            type(max_new_tokens) is not int or max_new_tokens < 1
        ):
            raise ValueError("At least one output slot is needed for the known anchor")

        def accepted(path: tuple[int, ...]) -> int:
            count = 0
            for parent, child in zip(path, path[1:]):
                if tokens[parent] in eos or predictions[parent] != tokens[child]:
                    break
                count += 1
            return count

        a0, a1 = accepted(self.original_nodes), accepted(self.refined_nodes)
        branch, path, count = ("original", self.original_nodes, a0)
        if a1 > a0:
            branch, path, count = "refined", self.refined_nodes, a1
        nodes = path[:count + 1]
        if max_new_tokens is not None:
            nodes = nodes[:max_new_tokens]
        committed = tuple(tokens[n] for n in nodes)
        bonus = None
        if committed[-1] not in eos and (
            max_new_tokens is None or len(committed) < max_new_tokens
        ):
            bonus = predictions[nodes[-1]]
        return GreedyDecision(branch, a0, a1, nodes, committed, bonus)

    def compaction_indices(
        self, decision: GreedyDecision, past_length: int
    ) -> tuple[int, ...]:
        """Physical cache source slots, materialized before any overlapping copy.

        The bonus has no target cache row yet and MUST NOT be gathered here.
        """
        self.positions(past_length)
        path = self.original_nodes if decision.branch == "original" else self.refined_nodes
        if decision.branch not in {"original", "refined"} or (
            not decision.accepted_nodes
            or path[:len(decision.accepted_nodes)] != decision.accepted_nodes
        ):
            raise ValueError("Decision is not an accepted prefix of this layout")
        return tuple(past_length + n for n in decision.accepted_nodes)
