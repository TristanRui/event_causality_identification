from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


RELATION_TYPES = ["NONE", "CAUSE", "TEMPORAL", "TREAT", "VERIFY"]
RELATION_ID_TO_TYPE = {idx: name for idx, name in enumerate(RELATION_TYPES)}

OBJECTIVES = [
    "root_cause_tracing",
    "action_recovery",
    "verification_closure",
]

OBJECTIVE_LABELS = {
    "root_cause_tracing": "Root cause tracing",
    "action_recovery": "Action recovery",
    "verification_closure": "Verification closure",
}

DEFAULT_PREDICTION_ROOT = Path("outputs") / "graph_predictions"
DEFAULT_OUTPUT_DIR = Path("outputs") / "process_level_utility"

Pair = Tuple[str, int, int]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_relation_type(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return RELATION_ID_TO_TYPE.get(value)
    return None


def node_type_lookup(nodes: Iterable[Dict[str, Any]], id_key: str) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for idx, node in enumerate(nodes):
        node_id = int(node.get(id_key, idx))
        lookup[node_id] = str(node.get("type", ""))
    return lookup


def exact_pred_to_gold_map(node_match: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for item in node_match:
        if item.get("match_level") != "exact":
            continue
        matched = item.get("matched_gold_node_id")
        if matched is None:
            continue
        mapping[int(item["pred_node_id"])] = int(matched)
    return mapping


def objective_pair(
    doc_id: str,
    head_id: int,
    tail_id: int,
    rel_type: str,
    node_types: Dict[int, str],
) -> Optional[Tuple[str, Pair]]:
    """Map a typed MCPG edge to a process-level query-answer pair.

    The pair format is (doc_id, query_gold_node_id, answer_gold_node_id).
    """
    head_type = node_types.get(head_id)
    tail_type = node_types.get(tail_id)

    if (
        rel_type == "CAUSE"
        and head_type == "ROOT_CAUSE"
        and tail_type in {"PHENOMENON", "FAILURE"}
    ):
        return "root_cause_tracing", (doc_id, tail_id, head_id)

    if (
        rel_type == "TREAT"
        and head_type == "ACTION"
        and tail_type in {"PHENOMENON", "FAILURE", "ROOT_CAUSE"}
    ):
        return "action_recovery", (doc_id, tail_id, head_id)

    if (
        rel_type == "VERIFY"
        and head_type == "VERIFICATION"
        and tail_type in {"ACTION", "PHENOMENON", "FAILURE"}
    ):
        return "verification_closure", (doc_id, tail_id, head_id)

    return None


def gold_pairs_from_edges(doc: Dict[str, Any]) -> Dict[str, Set[Pair]]:
    doc_id = str(doc.get("doc_id", ""))
    gold_types = node_type_lookup(doc.get("gold_nodes_ref", []) or [], "gold_node_id")
    pairs: Dict[str, Set[Pair]] = {name: set() for name in OBJECTIVES}

    for edge in doc.get("gold_edge_labels_ref", []) or []:
        rel_type = normalize_relation_type(edge.get("type"))
        if not rel_type:
            continue
        mapped = objective_pair(
            doc_id=doc_id,
            head_id=int(edge["head"]),
            tail_id=int(edge["tail"]),
            rel_type=rel_type,
            node_types=gold_types,
        )
        if mapped is None:
            continue
        objective, pair = mapped
        pairs[objective].add(pair)

    return pairs


def pred_pairs_from_edges(doc: Dict[str, Any]) -> Tuple[Dict[str, Set[Pair]], Counter]:
    doc_id = str(doc.get("doc_id", ""))
    gold_types = node_type_lookup(doc.get("gold_nodes_ref", []) or [], "gold_node_id")
    pred_to_gold = exact_pred_to_gold_map(doc.get("node_match", []) or [])
    pairs: Dict[str, Set[Pair]] = {name: set() for name in OBJECTIVES}
    diagnostics = Counter()

    for edge in doc.get("predicted_relations", []) or []:
        rel_type = normalize_relation_type(edge.get("type"))
        if not rel_type or rel_type == "NONE":
            continue
        diagnostics["predicted_positive_edges"] += 1

        pred_head = int(edge["head"])
        pred_tail = int(edge["tail"])
        if pred_head not in pred_to_gold or pred_tail not in pred_to_gold:
            diagnostics["unmapped_predicted_edges"] += 1
            continue

        gold_head = pred_to_gold[pred_head]
        gold_tail = pred_to_gold[pred_tail]
        mapped = objective_pair(
            doc_id=doc_id,
            head_id=gold_head,
            tail_id=gold_tail,
            rel_type=rel_type,
            node_types=gold_types,
        )
        if mapped is None:
            diagnostics["non_objective_predicted_edges"] += 1
            continue
        objective, pair = mapped
        pairs[objective].add(pair)

    return pairs, diagnostics


def metric_from_pairs(pred_pairs: Set[Pair], gold_pairs: Set[Pair]) -> Dict[str, Any]:
    correct_pairs = pred_pairs & gold_pairs
    precision = len(correct_pairs) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(correct_pairs) / len(gold_pairs) if gold_pairs else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "predicted": len(pred_pairs),
        "gold": len(gold_pairs),
        "correct": len(correct_pairs),
    }


def find_prediction_files(prediction_root: Path, pattern: str) -> List[Path]:
    if prediction_root.is_file():
        return [prediction_root]
    return sorted(prediction_root.glob(pattern))


def evaluate(prediction_files: List[Path]) -> Dict[str, Any]:
    total_gold: Dict[str, Set[Pair]] = {name: set() for name in OBJECTIVES}
    total_pred: Dict[str, Set[Pair]] = {name: set() for name in OBJECTIVES}
    per_doc_rows: List[Dict[str, Any]] = []
    diagnostics = Counter()

    for path in prediction_files:
        docs = load_jsonl(path)
        diagnostics["prediction_files"] += 1
        diagnostics["documents"] += len(docs)

        for doc in docs:
            doc_id = str(doc.get("doc_id", ""))
            gold_pairs = gold_pairs_from_edges(doc)
            pred_pairs, doc_diag = pred_pairs_from_edges(doc)
            diagnostics.update(doc_diag)

            for objective in OBJECTIVES:
                total_gold[objective].update(gold_pairs[objective])
                total_pred[objective].update(pred_pairs[objective])
                row_metrics = metric_from_pairs(pred_pairs[objective], gold_pairs[objective])
                per_doc_rows.append(
                    {
                        "doc_id": doc_id,
                        "objective": objective,
                        "objective_label": OBJECTIVE_LABELS[objective],
                        **row_metrics,
                    }
                )

    metrics = {
        objective: {
            "label": OBJECTIVE_LABELS[objective],
            **metric_from_pairs(total_pred[objective], total_gold[objective]),
        }
        for objective in OBJECTIVES
    }

    macro_f1 = sum(metrics[obj]["f1"] for obj in OBJECTIVES) / len(OBJECTIVES)
    macro_precision = sum(metrics[obj]["precision"] for obj in OBJECTIVES) / len(OBJECTIVES)
    macro_recall = sum(metrics[obj]["recall"] for obj in OBJECTIVES) / len(OBJECTIVES)

    return {
        "metrics": metrics,
        "macro_average": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "diagnostics": dict(diagnostics),
        "prediction_files": [str(path) for path in prediction_files],
        "per_document": per_doc_rows,
    }


def write_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["objective", "precision", "recall", "f1", "predicted", "gold", "correct"],
        )
        writer.writeheader()
        for objective in OBJECTIVES:
            item = metrics[objective]
            writer.writerow(
                {
                    "objective": item["label"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "predicted": item["predicted"],
                    "gold": item["gold"],
                    "correct": item["correct"],
                }
            )


def write_per_doc_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "objective",
                "objective_label",
                "precision",
                "recall",
                "f1",
                "predicted",
                "gold",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate process-level maintenance utility from decoded MCPG predictions."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help="Decoded graph prediction directory or one graph_predictions.jsonl file.",
    )
    parser.add_argument(
        "--pattern",
        default="fold_*/test/graph_predictions.jsonl",
        help="Glob pattern under --prediction-root when it is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for process_level_metrics.json/csv outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_files = find_prediction_files(args.prediction_root, args.pattern)
    if not prediction_files:
        raise FileNotFoundError(
            f"No graph_predictions.jsonl files found under {args.prediction_root} with pattern {args.pattern}"
        )

    result = evaluate(prediction_files)

    output_dir = args.output_dir
    save_json(result, output_dir / "process_level_metrics.json")
    write_metrics_csv(result["metrics"], output_dir / "process_level_metrics.csv")
    write_per_doc_csv(result["per_document"], output_dir / "process_level_metrics_by_doc.csv")

    print("\n================ PROCESS-LEVEL MAINTENANCE UTILITY ================")
    for objective in OBJECTIVES:
        item = result["metrics"][objective]
        print(
            f"{item['label']}: "
            f"P={item['precision']:.4f}, R={item['recall']:.4f}, F1={item['f1']:.4f} "
            f"(pred={item['predicted']}, gold={item['gold']}, correct={item['correct']})"
        )
    macro = result["macro_average"]
    print(f"Macro average: P={macro['precision']:.4f}, R={macro['recall']:.4f}, F1={macro['f1']:.4f}")
    print(f"saved json: {output_dir / 'process_level_metrics.json'}")
    print(f"saved csv:  {output_dir / 'process_level_metrics.csv'}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
