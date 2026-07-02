from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

GRAPH_SCHEMA_NAME = "unified_graph_sample"
DEFAULT_IGNORE_LABEL = -1

TARGET_KEYS = (
    "train",
    "strict",
    "strict_exact",
    "train_strict_exact_unmatched_none",
    "train_strict_exact_unmatched_ignore",
)


class GraphSchemaError(ValueError):
    """Raised when a graph sample cannot be adapted to the unified schema."""



def _copy_matrix(matrix: Optional[List[List[int]]]) -> Optional[List[List[int]]]:
    if matrix is None:
        return None
    return [list(row) for row in matrix]



def _ensure_dict(sample: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise GraphSchemaError(f"graph sample must be dict, got {type(sample)}")
    return sample



def _extract_ignore_label(sample: Dict[str, Any]) -> int:
    schema = sample.get("schema") or {}
    ignore_label = schema.get("ignore_label", sample.get("ignore_label", DEFAULT_IGNORE_LABEL))
    try:
        return int(ignore_label)
    except Exception as exc:
        raise GraphSchemaError(f"invalid ignore_label: {ignore_label}") from exc



def _base_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = _ensure_dict(sample)
    adapted = {
        "graph_schema": GRAPH_SCHEMA_NAME,
        "doc_id": str(sample.get("doc_id", "")),
        "title": sample.get("title", ""),
        "category": sample.get("category", ""),
        "text": sample.get("text", ""),
        "fold_index": sample.get("fold_index"),
        "split": sample.get("split"),
        "schema": deepcopy(sample.get("schema", {})),
        "nodes": deepcopy(sample.get("nodes", [])),
        "edge_labels": deepcopy(sample.get("edge_labels", [])),
        "role_constraint_mask": deepcopy(sample.get("role_constraint_mask", [])),
        "node_match": deepcopy(sample.get("node_match", [])),
        "gold_nodes_ref": deepcopy(sample.get("gold_nodes_ref", [])),
        "gold_edge_labels_ref": deepcopy(sample.get("gold_edge_labels_ref", [])),
        "meta": deepcopy(sample.get("meta", {})),
    }
    adapted["ignore_label"] = _extract_ignore_label(sample)
    return adapted



def _gold_targets(relation_matrix: List[List[int]]) -> Dict[str, Optional[List[List[int]]]]:
    """Gold graph samples use the same fully supervised matrix for train and strict evaluation."""
    return {key: _copy_matrix(relation_matrix) for key in TARGET_KEYS}



def _targets_from_predicted_sample(sample: Dict[str, Any]) -> Dict[str, Optional[List[List[int]]]]:
    """Collect strict target matrices from a predicted graph sample."""
    targets = sample.get("targets") or {}

    strict_matrix = targets.get(
        "strict_exact",
        targets.get("strict", sample.get("strict_exact_eval_relation_matrix", sample.get("strict_eval_relation_matrix"))),
    )

    train_matrix = targets.get("train", sample.get("relation_matrix"))

    normalized = {key: _copy_matrix(targets.get(key)) for key in TARGET_KEYS}
    normalized["train"] = _copy_matrix(train_matrix)
    normalized["strict"] = _copy_matrix(targets.get("strict", strict_matrix))
    normalized["strict_exact"] = _copy_matrix(strict_matrix)

    normalized["train_strict_exact_unmatched_none"] = _copy_matrix(
        targets.get("train_strict_exact_unmatched_none")
    )
    normalized["train_strict_exact_unmatched_ignore"] = _copy_matrix(
        targets.get("train_strict_exact_unmatched_ignore")
    )
    return normalized



def adapt_gold_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a gold graph sample into the unified graph schema.

    Gold samples are assumed to provide `relation_matrix` as both training and
    evaluation target. The unmatched none and unmatched ignore aliases also map
    to the gold relation matrix so graph training can use one target_key API.
    """
    adapted = _base_sample(sample)
    relation_matrix = sample.get("relation_matrix")
    if relation_matrix is None:
        raise GraphSchemaError("gold sample missing relation_matrix")

    adapted["targets"] = _gold_targets(relation_matrix)

    adapted["relation_matrix"] = _copy_matrix(relation_matrix)
    adapted["strict_eval_relation_matrix"] = _copy_matrix(relation_matrix)
    adapted["strict_exact_eval_relation_matrix"] = _copy_matrix(relation_matrix)
    adapted["meta"]["sample_origin"] = adapted["meta"].get("sample_origin", "gold")
    return adapted



def adapt_predicted_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a predicted graph sample into the unified graph schema.

    Predicted samples may not have a generic `train` target. They should provide
    a strict evaluation matrix and may provide strict noise-aware training targets
    under `targets`.
    """
    adapted = _base_sample(sample)
    adapted["targets"] = _targets_from_predicted_sample(sample)

    strict_matrix = adapted["targets"].get("strict_exact")
    train_matrix = adapted["targets"].get("train")

    adapted["relation_matrix"] = _copy_matrix(train_matrix)
    adapted["strict_eval_relation_matrix"] = _copy_matrix(strict_matrix)
    adapted["strict_exact_eval_relation_matrix"] = _copy_matrix(strict_matrix)
    adapted["meta"]["sample_origin"] = adapted["meta"].get("sample_origin", "predicted")
    return adapted



def adapt_graph_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-adapt a graph sample according to available label fields."""
    sample = _ensure_dict(sample)
    if "targets" in sample:
        adapted = _base_sample(sample)
        adapted["targets"] = _targets_from_predicted_sample(sample)

        adapted["relation_matrix"] = _copy_matrix(sample.get("relation_matrix", adapted["targets"].get("train")))
        adapted["strict_eval_relation_matrix"] = _copy_matrix(
            sample.get("strict_eval_relation_matrix", adapted["targets"].get("strict_exact"))
        )
        adapted["strict_exact_eval_relation_matrix"] = _copy_matrix(
            sample.get("strict_exact_eval_relation_matrix", adapted["targets"].get("strict_exact"))
        )
        return adapted

    if (
        sample.get("strict_eval_relation_matrix") is not None
        or sample.get("strict_exact_eval_relation_matrix") is not None
    ):
        return adapt_predicted_sample(sample)

    if sample.get("relation_matrix") is not None:
        return adapt_gold_sample(sample)

    raise GraphSchemaError("cannot infer graph sample type: missing relation_matrix and strict eval matrices")



def get_target_matrix(sample: Dict[str, Any], target_key: str = "train") -> Optional[List[List[int]]]:
    adapted = adapt_graph_sample(sample)
    if target_key not in TARGET_KEYS:
        raise GraphSchemaError(f"unsupported target_key: {target_key}")
    return _copy_matrix(adapted["targets"].get(target_key))



def has_target(sample: Dict[str, Any], target_key: str = "train") -> bool:
    matrix = get_target_matrix(sample, target_key=target_key)
    return matrix is not None



def describe_graph_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    adapted = adapt_graph_sample(sample)
    node_count = len(adapted.get("nodes", []))
    targets = adapted.get("targets", {})
    target_availability = {key: targets.get(key) is not None for key in TARGET_KEYS}
    return {
        "graph_schema": adapted.get("graph_schema"),
        "doc_id": adapted.get("doc_id"),
        "split": adapted.get("split"),
        "fold_index": adapted.get("fold_index"),
        "node_count": node_count,
        "has_train_target": targets.get("train") is not None,
        "has_strict_target": targets.get("strict") is not None,
        "target_availability": target_availability,
        "ignore_label": adapted.get("ignore_label", DEFAULT_IGNORE_LABEL),
        "sample_origin": adapted.get("meta", {}).get("sample_origin", "unknown"),
    }
