"""Training/data primitives for the conditional-loss comparison. No cloud API."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import math
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from conditional_redraft_model import FrozenDraft, ConditionalRefiner, DraftState, make_anchored_example
from conditional_redraft_objectives import conditional_objective, branch_progress
from conditional_redraft_decode import _features

from conditional_redraft_learning_config import CONFIG


def check_deadline(deadline, reserve=60):
    if time.time() + reserve >= deadline:
        raise TimeoutError("Learning attempt deadline; preserve partial evidence")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def choose_anchors(prompt_length, total_length, count=32):
    """Outcome-independent, evenly spaced response starts; no q0-round bias."""
    # Each anchor must have a visible future token AND an editable one.
    options = list(range(prompt_length, total_length - 2))
    if len(options) <= count:
        return options
    return [options[(i * (len(options) - 1)) // (count - 1)] for i in range(count)]


@torch.no_grad()
def collect_prompt(target, ordinary, prompt, deadline, config=CONFIG):
    """Native one-token greedy labels and aligned actual target hidden states."""
    device = target.device
    tokens = torch.tensor([prompt["input_ids"]], device=device)
    if not 1 <= tokens.shape[1] <= config["max_prompt_tokens"]:
        raise ValueError("Prompt outside frozen token length")
    started = time.perf_counter()
    result = target(tokens, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    cache = result.past_key_values
    features = [_features(result.hidden_states, ordinary.drafter.target_layer_ids)]
    next_token = int(result.logits[0, -1].argmax())
    del result
    generated = []
    for index in range(config["max_new_tokens"]):
        check_deadline(deadline)
        generated.append(next_token)
        token = tokens.new_tensor([[next_token]])
        result = target(token, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, logits_to_keep=1)
        features.append(_features(result.hidden_states, ordinary.drafter.target_layer_ids))
        if not torch.isfinite(result.logits).all():
            raise ValueError("Nonfinite native scores")
        if next_token in prompt["stop_token_ids"]:
            del result
            break
        next_token = int(result.logits[0, -1].argmax())
        del result
    ids = tokens.new_tensor(prompt["input_ids"] + generated)
    raw = torch.cat(features, dim=1)[0]
    del features, cache
    if len(raw) != len(ids):
        raise ValueError("Native teacher feature/token alignment failed")
    anchors = choose_anchors(len(prompt["input_ids"]), len(ids), config["max_anchors_per_prompt"])
    records, midpoints = [], []
    for anchor in anchors:
        check_deadline(deadline)
        example = make_anchored_example(ids, raw, anchor=anchor,
                                       block_size=config["block_size"], mask_token_id=ordinary.drafter.mask_token_id)
        state = ordinary.capture(noise_embedding=target.get_input_embeddings()(example.noise_ids),
                                 target_hidden=example.context_features, position_ids=example.positions)
        original = example.noise_ids.clone()
        original[:, 1:] = target.get_output_embeddings()(state.normalized[:, 1:]).argmax(-1)
        records.append(dict(anchor=anchor, original=original[0].cpu().tolist(),
                            labels=example.labels[0].cpu().tolist(), valid=example.valid[0].cpu().tolist()))
        midpoints.append(state.midpoint[0].cpu())
        del state, example
    sync(device)
    bank = dict(ids=ids.cpu(), features=raw.cpu(),
                midpoints=torch.stack(midpoints) if midpoints else torch.empty(0, config["block_size"], ordinary.drafter.config.hidden_size),
                prompt_length=len(prompt["input_ids"]))
    metadata = dict(id=prompt["id"], domain=prompt["domain"], split=prompt["split"],
                    input_tokens=len(prompt["input_ids"]), generated_tokens=len(generated),
                    capped=len(generated) == config["max_new_tokens"] and generated[-1] not in prompt["stop_token_ids"],
                    output_ids=generated, anchors=records, collection_seconds=time.perf_counter() - started,
                    label_source="actual same-shape native one-token greedy generation",
                    teacher_features="actual native generation states; no offline relabeling")
    return bank, metadata


@torch.no_grad()
def restore_example(target, ordinary, bank, metadata, index):
    """Reconstruct cached midpoint inputs using the exact original prefix slice."""
    device = target.device
    record = metadata["anchors"][index]
    anchor = record["anchor"]
    noise_ids = bank["ids"].new_full((1, ordinary.drafter.block_size), ordinary.drafter.mask_token_id).to(device)
    noise_ids[0, 0] = bank["ids"][anchor]
    noise = target.get_input_embeddings()(noise_ids)
    positions = torch.arange(anchor + noise.shape[1], device=device)[None]
    raw = bank["features"][:anchor].to(device)[None]
    context = ordinary.drafter.hidden_norm(ordinary.drafter.fc(raw))
    midpoint = bank["midpoints"][index:index + 1].to(device)
    state = DraftState(noise=noise, projected_context=context, positions=positions,
                       rotary=ordinary.drafter.rotary_emb(noise, positions), midpoint=midpoint,
                       final=midpoint, normalized=midpoint, split=ordinary.split, producer=id(ordinary),
                       context_past=0, attention_mask=None, shared_lower_context=None)
    original = torch.tensor([record["original"]], dtype=torch.long, device=device)
    labels = torch.tensor([record["labels"]], dtype=torch.long, device=device)
    valid = torch.tensor([record["valid"]], dtype=torch.bool, device=device)
    prefix = target.get_input_embeddings()(original[:, 1:2])
    return state, prefix, original, labels, valid


def ordered_examples(metadata, epoch, seed):
    rng = random.Random(seed + 1009 * epoch)
    order = list(range(len(metadata)))
    rng.shuffle(order)
    result = []
    # Grouping by prompt bounds host/device feature traffic. All arms get the
    # exact same order, and loss normalization spans the full effective batch.
    for prompt_index in order:
        anchor_order = list(range(len(metadata[prompt_index]["anchors"])))
        rng.shuffle(anchor_order)
        result.extend((prompt_index, anchor_index) for anchor_index in anchor_order)
    return result


def optimizer_step(model, optimizer, denominator, step, total_steps, config=CONFIG):
    if denominator <= 0:
        optimizer.zero_grad(set_to_none=True)
        return dict(updated=False, denominator=denominator)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite training gradient")
    warmup = max(1, math.ceil(total_steps * config["warmup_fraction"]))
    if step < warmup:
        multiplier = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup - 1)
        multiplier = 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"] * multiplier
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(updated=True, denominator=denominator, gradient_norm=float(norm),
                learning_rate=config["learning_rate"] * multiplier)


@torch.no_grad()
def evaluate(target, ordinary, trained, metadata, load_bank, deadline, config=CONFIG):
    """Evaluate a BF16 COPY: never quantize live FP32 optimizer parameters."""
    dtype = torch.bfloat16 if target.device.type == "cuda" else next(ordinary.parameters()).dtype
    deployed = ConditionalRefiner(ordinary, config["mode"]).to(dtype=dtype).eval()
    deployed.load_state_dict(trained.state_dict(), strict=True)
    weights_digest = state_digest(deployed.state_dict())
    rows = []
    for meta in metadata:
        check_deadline(deadline)
        bank = load_bank(meta["id"])
        sums = dict(original_progress=0, union_progress=0, alternative_progress=0,
                    repaired_blocks=0, damaged_alternatives=0, fully_correct_original=0,
                    damaged_fully_correct_alternatives=0, unreachable=0, blocks=0)
        details = []
        for index in range(len(meta["anchors"])):
            state, prefix, original, labels, valid = restore_example(target, ordinary, bank, meta, index)
            hidden = deployed(state, prefix)
            alternative = original.clone()
            alternative[:, 2:] = target.get_output_embeddings()(hidden[:, 2:]).argmax(-1)
            metrics = branch_progress(original, alternative, labels, valid)
            a0, a1 = int(metrics["original"][0]), int(metrics["alternative"][0])
            full = a0 == int(valid.sum()) - 1
            sums["original_progress"] += 1 + a0
            sums["union_progress"] += 1 + max(a0, a1)
            sums["alternative_progress"] += 1 + a1
            sums["repaired_blocks"] += int(a1 > a0)
            sums["damaged_alternatives"] += int(a1 < a0)
            sums["fully_correct_original"] += int(full)
            sums["damaged_fully_correct_alternatives"] += int(full and a1 < a0)
            sums["unreachable"] += int(a0 < 1)
            sums["blocks"] += 1
            details.append(dict(anchor=meta["anchors"][index]["anchor"], original=a0, alternative=a1,
                                union=max(a0, a1), future_capacity=int(valid.sum()) - 1,
                                alternative_tokens=alternative[0].cpu().tolist()))
        rows.append(dict(id=meta["id"], domain=meta["domain"], **sums, details=details,
                         weights_sha256=weights_digest))
    del deployed
    return rows
