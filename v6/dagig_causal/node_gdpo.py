"""Core finite-group math for iid old-policy Node-GDPO."""

from __future__ import annotations

import torch


def normalized_mass(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 1 or weights.numel() < 2:
        raise ValueError("Node-GDPO weights must be a one-dimensional action group")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("Node-GDPO weights must be finite and positive")
    return weights / weights.sum()


def bayesian_target(
    weights: torch.Tensor,
    pointwise_ig: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return q_beta(k) proportional to w_k exp(beta * IG_k)."""

    mass = normalized_mass(weights)
    if pointwise_ig.shape != mass.shape or not torch.isfinite(pointwise_ig).all():
        raise ValueError("Invalid Node-GDPO information-gain vector")
    return torch.softmax(torch.log(mass) + float(beta) * pointwise_ig, dim=0)


def importance_group_policy(
    weights: torch.Tensor,
    current_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Project pi_theta/pi_old onto iid samples without double weighting."""

    mass = normalized_mass(weights)
    if current_logprob.shape != mass.shape or old_logprob.shape != mass.shape:
        raise ValueError("Node-GDPO log-probabilities do not match the action group")
    if not torch.isfinite(current_logprob).all() or not torch.isfinite(old_logprob).all():
        raise ValueError("Node-GDPO log-probabilities must be finite")
    if temperature <= 0:
        raise ValueError("Node-GDPO temperature must be positive")
    log_ratio = (current_logprob - old_logprob) / float(temperature)
    return torch.softmax(torch.log(mass) + log_ratio, dim=0)


def forward_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("KL distributions must have identical shape")
    tiny = torch.finfo(left.dtype).tiny
    left_safe = left.clamp_min(tiny)
    right_safe = right.clamp_min(tiny)
    return torch.sum(left_safe * (torch.log(left_safe) - torch.log(right_safe)))


def gdpo_loss(
    weights: torch.Tensor,
    target: torch.Tensor,
    current_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy to the Bayesian target plus empirical old-policy KL."""

    mass = normalized_mass(weights)
    current = importance_group_policy(
        mass, current_logprob, old_logprob, temperature=temperature
    )
    target = normalized_mass(target)
    log_current = torch.log(current.clamp_min(torch.finfo(current.dtype).tiny))
    cross_entropy = -torch.sum(target * log_current)
    trust_kl = forward_kl(mass, current)
    loss = cross_entropy + float(kl_weight) * trust_kl
    return loss, {
        "cross_entropy": cross_entropy,
        "trust_kl": trust_kl,
        "current_distribution": current,
        "target_distribution": target,
    }


def processed_decode_sequence_logprobs(
    next_token_logits: torch.Tensor,
    prefix_input_ids: torch.Tensor,
    action_input_ids: torch.Tensor,
    action_lengths: torch.Tensor,
    *,
    repetition_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score teacher-forced actions under the frozen sampling distribution.

    `next_token_logits[:, j]` predicts `action_input_ids[:, j]`. Prefixes must
    have equal length within the group, as they do for same-parent actions.
    """

    if next_token_logits.ndim != 3 or action_input_ids.ndim != 2:
        raise ValueError("Invalid decode-policy tensor rank")
    batch, steps, _ = next_token_logits.shape
    if action_input_ids.shape != (batch, steps):
        raise ValueError("Action IDs do not align with next-token logits")
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != batch:
        raise ValueError("Prefix IDs do not align with action batch")
    if action_lengths.shape != (batch,):
        raise ValueError("Action lengths do not align with action batch")
    if repetition_penalty <= 0 or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid frozen decoding-policy parameter")

    per_action: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    for step in range(steps):
        active = torch.nonzero(action_lengths > step, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        scores = next_token_logits[active, step, :].float()
        if repetition_penalty != 1.0:
            history = torch.cat(
                [
                    prefix_input_ids[active],
                    action_input_ids[active, :step],
                ],
                dim=1,
            )
            seen_scores = torch.gather(scores, 1, history)
            adjusted = torch.where(
                seen_scores < 0,
                seen_scores * float(repetition_penalty),
                seen_scores / float(repetition_penalty),
            )
            scores = scores.scatter(1, history, adjusted)
        scores = scores / float(temperature)
        if top_k > 0:
            keep = min(int(top_k), scores.shape[-1])
            threshold = torch.topk(scores, keep, dim=-1).values[:, -1, None]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False, dim=-1)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative <= (1.0 - float(top_p))
            sorted_remove[:, -1:] = False
            remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
            scores = scores.masked_fill(remove, -torch.inf)
        log_probability = torch.log_softmax(scores, dim=-1)
        targets = action_input_ids[active, step]
        selected = torch.gather(log_probability, 1, targets[:, None]).squeeze(1)
        for local_index, batch_index in enumerate(active.tolist()):
            per_action[batch_index].append(selected[local_index])

    sums: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    for values in per_action:
        if not values:
            raise ValueError("Empty credited node action")
        stacked = torch.stack(values)
        sums.append(stacked.sum())
        means.append(stacked.mean())
    return torch.stack(sums), torch.stack(means)
