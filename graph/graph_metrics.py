from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Any

import torch
import torch.nn as nn

from graph.graph_core import (
    RELATION_TYPES,
    RELATION_TYPE_TO_ID,
    RELATION_ID_TO_TYPE,
    apply_hard_constraints_and_decode,
    compute_violation_count,
)

EVAL_POSITIVE_MARGIN = float(os.environ.get("RELATION_EVAL_POSITIVE_MARGIN", os.environ.get("RELATION_POSITIVE_MARGIN", "0.5")))
EVAL_POSITIVE_MIN_PROB = float(os.environ.get("RELATION_EVAL_POSITIVE_MIN_PROB", "0.0"))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _as_ignore_label_list(ignore_labels: Optional[Sequence[int]], size: int) -> List[int]:
    if ignore_labels is None:
        return [-1 for _ in range(size)]
    if isinstance(ignore_labels, torch.Tensor):
        return [int(x.item()) for x in ignore_labels]
    if len(ignore_labels) == size:
        return [int(x) for x in ignore_labels]
    if len(ignore_labels) == 1:
        return [int(ignore_labels[0]) for _ in range(size)]
    raise ValueError(f"ignore_labels 长度 {len(ignore_labels)} 与 batch 大小 {size} 不一致")


def apply_positive_eval_filter(pred_matrix: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    none_id = RELATION_TYPE_TO_ID["NONE"]
    pred_scores = probs.gather(-1, pred_matrix.unsqueeze(-1)).squeeze(-1)
    pred_is_positive = pred_matrix != none_id
    weak_positive = torch.zeros_like(pred_is_positive, dtype=torch.bool)

    if EVAL_POSITIVE_MARGIN > 0:
        none_scores = probs[..., none_id]
        weak_positive = weak_positive | (pred_is_positive & ((pred_scores - none_scores) < EVAL_POSITIVE_MARGIN))
    if EVAL_POSITIVE_MIN_PROB > 0:
        weak_positive = weak_positive | (pred_is_positive & (pred_scores < EVAL_POSITIVE_MIN_PROB))

    if weak_positive.any():
        pred_matrix = pred_matrix.clone()
        pred_matrix[weak_positive] = none_id
        pred_scores = probs.gather(-1, pred_matrix.unsqueeze(-1)).squeeze(-1)
    return pred_matrix, pred_scores


def build_relation_class_weights(dataset) -> torch.Tensor:
    """Compute class weights from the active target matrix in the dataset."""
    counts = torch.zeros(len(RELATION_TYPES), dtype=torch.float)
    for feature in dataset.features:
        labels = feature.get("target_matrix")
        ignore_label = int(feature.get("ignore_label", -1))
        if labels is None:
            continue
        node_count = labels.size(0)
        for i in range(node_count):
            for j in range(node_count):
                if i == j:
                    continue
                label = int(labels[i, j].item())
                if label < 0 or label == ignore_label:
                    continue
                counts[label] += 1

    counts = counts.clamp(min=1.0)
    inv = counts.sum() / counts
    inv = inv / inv.mean()

    # With predicted nodes, many unmatched pairs are ignored. The remaining
    # valid target matrix can make the natural inverse-frequency NONE weight
    # extremely small, so false positives become almost free. Keep the floor
    # configurable because this is the main precision/recall knob.
    none_id = RELATION_TYPE_TO_ID["NONE"]
    none_floor = float(os.environ.get("RELATION_NONE_CLASS_WEIGHT_FLOOR", "0.10"))
    none_cap = float(os.environ.get("RELATION_NONE_CLASS_WEIGHT_CAP", "0"))
    if none_floor > 0:
        inv[none_id] = max(float(inv[none_id]), none_floor)
    if none_cap > 0:
        inv[none_id] = min(float(inv[none_id]), none_cap)

    raw_overrides = os.environ.get("RELATION_CLASS_WEIGHT_OVERRIDES", "").strip()
    if raw_overrides:
        for item in raw_overrides.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"RELATION_CLASS_WEIGHT_OVERRIDES item must be TYPE:VALUE, got {item!r}")
            rel_type, value = item.split(":", 1)
            rel_type = rel_type.strip()
            if rel_type not in RELATION_TYPE_TO_ID:
                raise ValueError(f"Unknown relation type in RELATION_CLASS_WEIGHT_OVERRIDES: {rel_type}")
            inv[RELATION_TYPE_TO_ID[rel_type]] = float(value)
    return inv


def compute_relation_loss(
    pair_logits: List[torch.Tensor],
    target_matrices: List[Optional[torch.Tensor]],
    role_constraint_masks: List[torch.Tensor],
    class_weights: torch.Tensor,
    ignore_labels: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute relation classification loss with optional targets and IGNORE labels."""
    if device is None:
        device = class_weights.device

    total_loss = torch.tensor(0.0, device=device)
    valid_docs = 0
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")
    ignore_label_list = _as_ignore_label_list(ignore_labels, len(pair_logits))

    for doc_idx, (logits, labels, role_mask) in enumerate(zip(pair_logits, target_matrices, role_constraint_masks)):
        if labels is None or logits.numel() == 0:
            continue

        labels = labels.to(device)
        role_mask = role_mask.to(device)
        ignore_label = ignore_label_list[doc_idx]

        node_count = min(logits.size(0), labels.size(0), role_mask.size(0))
        logits = logits[:node_count, :node_count, :]
        labels = labels[:node_count, :node_count]
        role_mask = role_mask[:node_count, :node_count, :]
        masked_logits = logits.masked_fill(role_mask <= 0, -1e4)

        valid_pairs = []
        valid_targets = []
        for i in range(node_count):
            for j in range(node_count):
                if i == j:
                    continue
                label = int(labels[i, j].item())
                if label < 0 or label == ignore_label:
                    continue
                if label >= role_mask.size(-1) or float(role_mask[i, j, label].item()) <= 0:
                    continue
                valid_pairs.append(masked_logits[i, j])
                valid_targets.append(labels[i, j])

        if not valid_pairs:
            continue

        valid_pairs_tensor = torch.stack(valid_pairs, dim=0)
        valid_targets_tensor = torch.stack(valid_targets, dim=0)
        total_loss = total_loss + loss_fn(valid_pairs_tensor, valid_targets_tensor)
        valid_docs += 1

    return total_loss / max(valid_docs, 1)


def extract_positive_edges_from_matrix(
    relation_matrix: torch.Tensor,
    ignore_label: int = -1,
) -> List[Tuple[int, int, str]]:
    edges: List[Tuple[int, int, str]] = []
    if relation_matrix is None or relation_matrix.numel() == 0:
        return edges

    node_count = relation_matrix.size(0)
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            rel_id = int(relation_matrix[i, j].item())
            if rel_id < 0 or rel_id == ignore_label:
                continue
            if rel_id == RELATION_TYPE_TO_ID["NONE"]:
                continue
            edges.append((i, j, RELATION_ID_TO_TYPE[rel_id]))
    return edges


def compute_relation_metrics(
    pred_relation_matrices: List[torch.Tensor],
    gold_relation_matrices: List[torch.Tensor],
    ignore_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    pred_total = 0
    gold_total = 0
    correct_total = 0
    per_type = {name: {"pred": 0, "gold": 0, "correct": 0} for name in RELATION_TYPES if name != "NONE"}

    ignore_label_list = _as_ignore_label_list(ignore_labels, len(pred_relation_matrices))

    none_id = RELATION_TYPE_TO_ID["NONE"]
    for pred_matrix, gold_matrix, ignore_label in zip(pred_relation_matrices, gold_relation_matrices, ignore_label_list):
        pred_matrix = pred_matrix.cpu()
        gold_matrix = gold_matrix.cpu()
        node_count = min(pred_matrix.size(0), gold_matrix.size(0))

        for i in range(node_count):
            for j in range(node_count):
                if i == j:
                    continue

                gold_id = int(gold_matrix[i, j].item())
                if gold_id < 0 or gold_id == ignore_label:
                    continue

                pred_id = int(pred_matrix[i, j].item())
                if pred_id < 0 or pred_id == ignore_label:
                    pred_id = none_id

                gold_is_positive = gold_id != none_id and gold_id in RELATION_ID_TO_TYPE
                pred_is_positive = pred_id != none_id and pred_id in RELATION_ID_TO_TYPE

                if gold_is_positive:
                    gold_total += 1
                    per_type[RELATION_ID_TO_TYPE[gold_id]]["gold"] += 1
                if pred_is_positive:
                    pred_total += 1
                    per_type[RELATION_ID_TO_TYPE[pred_id]]["pred"] += 1
                if gold_is_positive and pred_id == gold_id:
                    correct_total += 1
                    per_type[RELATION_ID_TO_TYPE[gold_id]]["correct"] += 1

    precision = safe_div(correct_total, pred_total)
    recall = safe_div(correct_total, gold_total)
    f1 = safe_div(2 * precision * recall, precision + recall)

    per_type_metrics = summarize_per_relation_type(per_type)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_type_metrics": per_type_metrics,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "correct_total": correct_total,
    }


def summarize_per_relation_type(per_type_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for rel_type, stat in per_type_stats.items():
        precision = safe_div(stat["correct"], stat["pred"])
        recall = safe_div(stat["correct"], stat["gold"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        summary[rel_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred": int(stat["pred"]),
            "gold": int(stat["gold"]),
            "correct": int(stat["correct"]),
        }
    return summary


def compute_graph_confidence(
    pred_relation_matrix: torch.Tensor,
    pred_score_matrix: torch.Tensor,
    ignore_label: int = -1,
) -> float:
    scores: List[float] = []
    node_count = pred_relation_matrix.size(0)
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            rel_id = int(pred_relation_matrix[i, j].item())
            if rel_id < 0 or rel_id == ignore_label:
                continue
            if rel_id == RELATION_TYPE_TO_ID["NONE"]:
                continue
            scores.append(float(pred_score_matrix[i, j].item()))
    return sum(scores) / len(scores) if scores else 0.0


def collect_edge_confidences(
    pred_relation_matrix: torch.Tensor,
    pred_score_matrix: torch.Tensor,
    gold_relation_matrix: torch.Tensor,
    ignore_label: int = -1,
) -> Tuple[List[float], List[int]]:
    confidences: List[float] = []
    correctness: List[int] = []
    node_count = pred_relation_matrix.size(0)

    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            pred_rel = int(pred_relation_matrix[i, j].item())
            gold_rel = int(gold_relation_matrix[i, j].item())
            if pred_rel < 0 or pred_rel == ignore_label:
                continue
            if gold_rel < 0 or gold_rel == ignore_label:
                continue
            if pred_rel == RELATION_TYPE_TO_ID["NONE"]:
                continue
            confidences.append(float(pred_score_matrix[i, j].item()))
            correctness.append(1 if pred_rel == gold_rel else 0)

    return confidences, correctness


def calibration_summary(confidences: List[float], correctness: List[int], num_bins: int = 10) -> Dict[str, Any]:
    if not confidences:
        return {
            "num_bins": num_bins,
            "ece": 0.0,
            "brier": 0.0,
            "bin_stats": [],
        }

    total = len(confidences)
    ece = 0.0
    brier = 0.0
    bin_stats: List[Dict[str, Any]] = []

    for conf, corr in zip(confidences, correctness):
        brier += (conf - corr) ** 2
    brier /= total

    for bin_idx in range(num_bins):
        left = bin_idx / num_bins
        right = (bin_idx + 1) / num_bins
        if bin_idx == num_bins - 1:
            selected = [k for k, conf in enumerate(confidences) if left <= conf <= right]
        else:
            selected = [k for k, conf in enumerate(confidences) if left <= conf < right]

        if not selected:
            bin_stats.append({
                "bin": bin_idx,
                "left": left,
                "right": right,
                "count": 0,
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
            })
            continue

        avg_confidence = sum(confidences[k] for k in selected) / len(selected)
        avg_accuracy = sum(correctness[k] for k in selected) / len(selected)
        weight = len(selected) / total
        ece += abs(avg_confidence - avg_accuracy) * weight
        bin_stats.append({
            "bin": bin_idx,
            "left": left,
            "right": right,
            "count": len(selected),
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
        })

    return {
        "num_bins": num_bins,
        "ece": ece,
        "brier": brier,
        "bin_stats": bin_stats,
    }


def collect_model_outputs(
    model: nn.Module,
    dataloader,
    class_weights: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    pred_matrices: List[torch.Tensor] = []
    gold_matrices: List[torch.Tensor] = []
    used_ignore_labels: List[int] = []
    total_violations = 0
    graph_confidences: List[float] = []
    edge_confidences: List[float] = []
    edge_correctness: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            target_matrices = batch.get("target_matrices") or batch.get("relation_labels")
            ignore_labels = batch.get("ignore_labels")
            outputs = model(batch)
            loss = compute_relation_loss(
                outputs["pair_logits"],
                target_matrices,
                batch["role_constraint_masks"],
                class_weights,
                ignore_labels=ignore_labels,
                device=device,
            )
            total_loss += float(loss.item())

            ignore_label_list = _as_ignore_label_list(ignore_labels, len(outputs["pair_logits"]))
            for logits, gold_labels, role_mask, ignore_label in zip(
                outputs["pair_logits"],
                target_matrices,
                batch["role_constraint_masks"],
                ignore_label_list,
            ):
                if gold_labels is None:
                    continue
                node_count = min(logits.size(0), gold_labels.size(0), role_mask.size(0))
                logits = logits[:node_count, :node_count, :]
                gold_labels = gold_labels[:node_count, :node_count]
                role_mask = role_mask[:node_count, :node_count, :]

                masked_logits = logits.masked_fill(role_mask.to(logits.device) <= 0, -1e4)
                probs = torch.softmax(masked_logits, dim=-1)
                pred_matrix = apply_hard_constraints_and_decode(logits, role_mask)
                pred_matrix, pred_scores = apply_positive_eval_filter(pred_matrix, probs)

                pred_matrices.append(pred_matrix.detach().cpu())
                gold_matrices.append(gold_labels.detach().cpu())
                used_ignore_labels.append(ignore_label)
                total_violations += compute_violation_count(pred_matrix.detach().cpu(), role_mask.detach().cpu())
                graph_confidences.append(compute_graph_confidence(pred_matrix.detach().cpu(), pred_scores.detach().cpu(), ignore_label))

                doc_confidences, doc_correctness = collect_edge_confidences(
                    pred_matrix.detach().cpu(),
                    pred_scores.detach().cpu(),
                    gold_labels.detach().cpu(),
                    ignore_label,
                )
                edge_confidences.extend(doc_confidences)
                edge_correctness.extend(doc_correctness)

    metrics = compute_relation_metrics(pred_matrices, gold_matrices, ignore_labels=used_ignore_labels)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    metrics["violation_count"] = total_violations
    metrics["eval_positive_margin"] = EVAL_POSITIVE_MARGIN
    metrics["eval_positive_min_prob"] = EVAL_POSITIVE_MIN_PROB
    metrics["avg_graph_confidence"] = sum(graph_confidences) / len(graph_confidences) if graph_confidences else 0.0
    metrics["calibration"] = calibration_summary(edge_confidences, edge_correctness)
    return metrics


def _resolve_doc_target_matrix(doc: Dict[str, Any], target_key: str) -> Optional[torch.Tensor]:
    if "targets" in doc and isinstance(doc["targets"], dict):
        matrix = doc["targets"].get(target_key)
        if matrix is not None:
            return torch.tensor(matrix, dtype=torch.long) if not isinstance(matrix, torch.Tensor) else matrix

    legacy_key_map = {
        "train": "relation_matrix",
        "strict": "strict_eval_relation_matrix",
        "strict_exact": "strict_exact_eval_relation_matrix",
    }
    legacy_key = legacy_key_map.get(target_key)
    if legacy_key and doc.get(legacy_key) is not None:
        matrix = doc[legacy_key]
        return torch.tensor(matrix, dtype=torch.long) if not isinstance(matrix, torch.Tensor) else matrix
    return None


def evaluate_decoded_docs(decoded_docs: List[Dict[str, Any]], target_key: str = "strict") -> Dict[str, Any]:
    pred_matrices: List[torch.Tensor] = []
    gold_matrices: List[torch.Tensor] = []
    ignore_labels: List[int] = []
    confidences: List[float] = []

    for doc in decoded_docs:
        pred_matrix = doc.get("pred_relation_matrix")
        if pred_matrix is None:
            continue
        if not isinstance(pred_matrix, torch.Tensor):
            pred_matrix = torch.tensor(pred_matrix, dtype=torch.long)

        gold_matrix = _resolve_doc_target_matrix(doc, target_key=target_key)
        if gold_matrix is None:
            continue

        ignore_label = int(doc.get("schema", {}).get("ignore_label", -1))
        pred_matrices.append(pred_matrix.cpu())
        gold_matrices.append(gold_matrix.cpu())
        ignore_labels.append(ignore_label)

        graph_conf = doc.get("graph_confidence")
        if graph_conf is not None:
            confidences.append(float(graph_conf))

    metrics = compute_relation_metrics(pred_matrices, gold_matrices, ignore_labels=ignore_labels)
    metrics["document_count"] = len(pred_matrices)
    metrics["avg_graph_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    return metrics


def _node_signature(node: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(node["start"]), int(node["end"]), str(node["type"]))


def _gold_node_signature_lookup(doc: Dict[str, Any]) -> Dict[int, Tuple[int, int, str]]:
    lookup: Dict[int, Tuple[int, int, str]] = {}
    for idx, node in enumerate(doc.get("gold_nodes_ref", []) or []):
        node_id = int(node.get("gold_node_id", idx))
        lookup[node_id] = _node_signature(node)
    return lookup


def _pred_node_signature_lookup(doc: Dict[str, Any]) -> Dict[int, Tuple[int, int, str]]:
    lookup: Dict[int, Tuple[int, int, str]] = {}
    for idx, node in enumerate(doc.get("nodes", []) or []):
        node_id = int(node.get("node_id", idx))
        lookup[node_id] = _node_signature(node)
    return lookup


def _gold_end_to_end_edges(doc: Dict[str, Any]) -> set:
    edges = set()
    for rel in doc.get("gold_edge_labels_ref", []) or []:
        head = int(rel["head"])
        tail = int(rel["tail"])
        rel_type = str(rel["type"])
        if rel_type != "NONE":
            edges.add((head, tail, rel_type))
    return edges


def _node_match_levels_for_target(target_key: str) -> set:
    if target_key in {"strict", "strict_exact"}:
        return {"exact"}
    if target_key in {"train_strict_exact_unmatched_none", "train_strict_exact_unmatched_ignore"}:
        return {"exact"}
    return {"exact"}


def _pred_to_gold_lookup(doc: Dict[str, Any], target_key: str) -> Dict[int, int]:
    allowed_levels = _node_match_levels_for_target(target_key)
    lookup: Dict[int, int] = {}
    for item in doc.get("node_match", []) or []:
        if item.get("matched_gold_node_id") is None:
            continue
        if item.get("match_level") in allowed_levels:
            lookup[int(item["pred_node_id"])] = int(item["matched_gold_node_id"])
    return lookup


def _pred_end_to_end_edges(doc: Dict[str, Any], target_key: str) -> Tuple[set, set]:
    pred_to_gold = _pred_to_gold_lookup(doc, target_key)
    edges = set()
    mapped_edges = set()
    for rel in doc.get("predicted_relations", []) or []:
        head = int(rel["head"])
        tail = int(rel["tail"])
        rel_type = str(rel["type"])
        if rel_type == "NONE":
            continue
        edge = (head, tail, rel_type)
        edges.add(edge)
        if head in pred_to_gold and tail in pred_to_gold:
            mapped_edges.add((pred_to_gold[head], pred_to_gold[tail], rel_type))
    return edges, mapped_edges


def compute_end_to_end_graph_metrics(decoded_docs: List[Dict[str, Any]], target_key: str = "strict_exact") -> Dict[str, Any]:
    pred_total = 0
    gold_total = 0
    correct_total = 0
    per_type = {name: {"pred": 0, "gold": 0, "correct": 0} for name in RELATION_TYPES if name != "NONE"}

    for doc in decoded_docs:
        pred_edges, mapped_pred_edges = _pred_end_to_end_edges(doc, target_key=target_key)
        gold_edges = _gold_end_to_end_edges(doc)
        correct_edges = mapped_pred_edges & gold_edges

        pred_total += len(pred_edges)
        gold_total += len(gold_edges)
        correct_total += len(correct_edges)
        for edge in pred_edges:
            per_type[edge[2]]["pred"] += 1
        for edge in gold_edges:
            per_type[edge[2]]["gold"] += 1
        for edge in correct_edges:
            per_type[edge[2]]["correct"] += 1

    precision = safe_div(correct_total, pred_total)
    recall = safe_div(correct_total, gold_total)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "correct_total": correct_total,
        "target_key": target_key,
        "per_type_metrics": summarize_per_relation_type(per_type),
    }
