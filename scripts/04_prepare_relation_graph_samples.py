from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    candidates = [current.parent] + list(current.parent.parents)
    for p in candidates:
        if (p / "config.py").exists() and (p / "utils").exists() and (p / "data").exists():
            return p
    return start.resolve().parents[1]


PROJECT_DIR = find_project_root(Path(__file__))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

GOLD_JSONL = PROJECT_DIR / "data" / "annotations" / "gold_sample.jsonl"
FOLD_JSON = PROJECT_DIR / "data" / "processed" / "event_document_folds.json"
DEFAULT_PREDICTION_ROOT = PROJECT_DIR / "outputs" / "event_extraction"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "relation_graph_samples"


def resolve_project_path(path_text: str) -> Path:
    """Resolve absolute paths directly and relative paths from the project root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


OUTPUT_DIR = resolve_project_path(os.environ.get("OUTPUT_GRAPH_SAMPLE_ROOT", str(DEFAULT_OUTPUT_DIR)))
SUMMARY_PATH = OUTPUT_DIR / "relation_graph_predicted_summary.json"

_RUN_FOLDS_ENV = os.environ.get("RUN_FOLDS", "0,1,2,3,4").strip()
RUN_FOLDS_FILTER = {
    int(item.strip())
    for item in _RUN_FOLDS_ENV.split(",")
    if item.strip()
} if _RUN_FOLDS_ENV else None

_INCLUDE_SPLITS_ENV = os.environ.get("INCLUDE_SPLITS", "train,dev,test").strip()
INCLUDE_SPLITS_FILTER = {
    item.strip().lower()
    for item in _INCLUDE_SPLITS_ENV.split(",")
    if item.strip()
} if _INCLUDE_SPLITS_ENV else None

EVENT_TYPES = ["PHENOMENON", "FAILURE", "ROOT_CAUSE", "ACTION", "VERIFICATION"]
RELATION_TYPES = ["NONE", "CAUSE", "TEMPORAL", "TREAT", "VERIFY"]
EVAL_RELATION_TYPES = ["IGNORE", "NONE", "CAUSE", "TEMPORAL", "TREAT", "VERIFY"]
RELATION_TO_ID = {name: idx for idx, name in enumerate(RELATION_TYPES)}
EVAL_RELATION_TO_ID = {name: idx - 1 for idx, name in enumerate(EVAL_RELATION_TYPES)}  # IGNORE=-1, NONE=0...

EXACT = "exact"
SAME_TYPE_OVERLAP = "same_type_overlap"
ANY_TYPE_OVERLAP = "any_type_overlap"
UNMATCHED = "unmatched"

SAME_TYPE_IOU_THRESHOLD = 0.5
ANY_TYPE_IOU_THRESHOLD = 0.5
GRAPH_SAMPLE_SCHEMA_NAME = "relation_graph_samples"
PREFERRED_NODE_SOURCE_FIELD = "predicted_nodes"
ALLOW_UNCHECKED_PREDICTIONS = os.environ.get("ALLOW_UNCHECKED_PREDICTIONS", "0").strip() == "1"
INCLUDE_OBSERVED_ROLE_TRIPLES = os.environ.get("INCLUDE_OBSERVED_ROLE_TRIPLES", "0").strip() == "1"
ROLE_CONSTRAINT_PROFILE = "strict_domain"

STRICT_CAUSE_TYPE_PAIRS = {
    ("ROOT_CAUSE", "FAILURE"),
    ("ROOT_CAUSE", "PHENOMENON"),
    ("FAILURE", "PHENOMENON"),
}

TREAT_TYPE_PAIRS = {
    ("ACTION", "PHENOMENON"),
    ("ACTION", "FAILURE"),
    ("ACTION", "ROOT_CAUSE"),
}

VERIFY_TYPE_PAIRS = {
    ("VERIFICATION", "ACTION"),
    ("VERIFICATION", "PHENOMENON"),
    ("VERIFICATION", "FAILURE"),
}

# TEMPORAL is retained only as a residual procedural-order relation under
# the strict domain profile described in the manuscript.
TEMPORAL_CANDIDATE_TYPE_PAIRS = {
    ("ACTION", "ACTION"),
    ("VERIFICATION", "VERIFICATION"),
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n+", text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        spans.append((0, len(text)))
    return spans


def build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in "。！？!?；;\n":
            end = i + 1
            if end > start:
                spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        spans.append((0, len(text)))
    return spans


def find_span_bucket(start: int, end: int, buckets: List[Tuple[int, int]]) -> int:
    center = (start + end) / 2.0
    for idx, (s, e) in enumerate(buckets):
        if s <= center <= e:
            return idx
    for idx, (s, e) in enumerate(buckets):
        if s <= start < e:
            return idx
    return max(0, len(buckets) - 1)


def overlap_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    denom = max(1, min(a_end - a_start, b_end - b_start))
    return inter / denom


def _normalize_predicted_node(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    start = item.get("start")
    if start is None:
        start = item.get("char_start")
    end = item.get("end")
    if end is None:
        end = item.get("char_end")
    text = item.get("text", "")
    ev_type = item.get("type")
    if start is None or end is None or ev_type not in EVENT_TYPES:
        return None

    return {
        "pred_event_id": item.get("pred_event_id") or f"P{idx + 1}",
        "node_id": int(item.get("node_id", idx)),
        "start": int(start),
        "end": int(end),
        "text": text,
        "type": ev_type,
        "score": float(item.get("score", 0.0) or 0.0),
        "support_count": int(item.get("support_count", 1) or 1),
        "view_consistency_score": float(item.get("view_consistency_score", 0.0) or 0.0),
        "support_views": list(item.get("support_views", []) or []),
        "section_distribution": dict(item.get("section_distribution", {}) or {}),
    }


def normalize_predicted_nodes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_sources = [
        (PREFERRED_NODE_SOURCE_FIELD, record.get(PREFERRED_NODE_SOURCE_FIELD)),
        ("predicted_spans", record.get("predicted_spans")),
        ("predicted_events", record.get("predicted_events")),
        ("predictions", record.get("predictions")),
    ]

    chosen_source = None
    raw_candidates: List[Dict[str, Any]] = []
    for source_name, source_value in node_sources:
        if source_value:
            chosen_source = source_name
            raw_candidates = list(source_value)
            break

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            continue
        node = _normalize_predicted_node(item, idx)
        if node is None:
            continue
        key = (node["start"], node["end"], node["type"], node["text"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(node)

    normalized.sort(key=lambda x: (x["start"], x["end"], x["type"]))
    for idx, item in enumerate(normalized):
        item["node_id"] = idx
        item["pred_event_id"] = f"P{idx + 1}"
        item["node_source_field"] = chosen_source or "unknown"
        item["node_quality"] = {
            "score": item["score"],
            "support_count": item["support_count"],
            "view_consistency_score": item["view_consistency_score"],
        }
    return normalized


def build_gold_nodes(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    text = normalize_text(doc["text"])
    sent_spans = build_sentence_spans(text)
    para_spans = build_paragraph_spans(text)
    events = list(doc.get("event_mentions", []))
    events.sort(key=lambda e: (e["start"], e["end"], e["event_id"]))
    nodes: List[Dict[str, Any]] = []
    event_id_to_node_id: Dict[str, int] = {}
    for idx, event in enumerate(events):
        node = {
            "gold_node_id": idx,
            "event_id": event["event_id"],
            "start": int(event["start"]),
            "end": int(event["end"]),
            "text": event["text"],
            "type": event["type"],
            "sent_id": find_span_bucket(int(event["start"]), int(event["end"]), sent_spans),
            "para_id": find_span_bucket(int(event["start"]), int(event["end"]), para_spans),
        }
        nodes.append(node)
        event_id_to_node_id[event["event_id"]] = idx
    return nodes, event_id_to_node_id


def build_gold_relation_lookup(doc: Dict[str, Any], event_id_to_node_id: Dict[str, int]) -> Dict[Tuple[int, int], str]:
    lookup: Dict[Tuple[int, int], str] = {}
    for rel in doc.get("relations", []):
        head = rel.get("head")
        tail = rel.get("tail")
        rel_type = rel.get("type")
        if head in event_id_to_node_id and tail in event_id_to_node_id and rel_type in RELATION_TO_ID:
            lookup[(event_id_to_node_id[head], event_id_to_node_id[tail])] = rel_type
    return lookup


def build_observed_role_triples(gold_docs: List[Dict[str, Any]]) -> set:
    triples = set()
    for doc in gold_docs:
        gold_nodes, event_id_to_node_id = build_gold_nodes(doc)
        node_by_id = {node["gold_node_id"]: node for node in gold_nodes}
        relation_lookup = build_gold_relation_lookup(doc, event_id_to_node_id)
        for (head, tail), rel_type in relation_lookup.items():
            if rel_type not in RELATION_TO_ID or head not in node_by_id or tail not in node_by_id:
                continue
            triples.add((node_by_id[head]["type"], node_by_id[tail]["type"], rel_type))
    return triples


def build_default_role_triples() -> set:
    """Build the strict manuscript role constraints for event-type and relation triples."""
    triples = set()
    cause_pairs = set(STRICT_CAUSE_TYPE_PAIRS)

    for head_t, tail_t in cause_pairs:
        triples.add((head_t, tail_t, "CAUSE"))

    for head_t, tail_t in TREAT_TYPE_PAIRS:
        triples.add((head_t, tail_t, "TREAT"))

    for head_t, tail_t in VERIFY_TYPE_PAIRS:
        triples.add((head_t, tail_t, "VERIFY"))

    specific_pairs = cause_pairs | TREAT_TYPE_PAIRS | VERIFY_TYPE_PAIRS
    temporal_pairs = TEMPORAL_CANDIDATE_TYPE_PAIRS - specific_pairs
    for head_t, tail_t in sorted(temporal_pairs):
        triples.add((head_t, tail_t, "TEMPORAL"))

    return triples


def build_role_constraint_mask(nodes: List[Dict[str, Any]], allowed_role_triples: Optional[set] = None) -> List[List[List[int]]]:
    if allowed_role_triples is None:
        allowed_role_triples = build_default_role_triples()
    n = len(nodes)
    num_rel = len(RELATION_TYPES)
    mask = [[[1 for _ in range(num_rel)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                for r in range(num_rel):
                    mask[i][j][r] = 0
                mask[i][j][RELATION_TO_ID["NONE"]] = 1
                continue
            head_t = nodes[i]["type"]
            tail_t = nodes[j]["type"]
            allowed = {"NONE"}
            for rel_name in RELATION_TYPES:
                if rel_name == "NONE":
                    continue
                if (head_t, tail_t, rel_name) in allowed_role_triples:
                    allowed.add(rel_name)
            for rel_name, rel_id in RELATION_TO_ID.items():
                mask[i][j][rel_id] = 1 if rel_name in allowed else 0
    return mask


def generate_match_candidates(pred_nodes: List[Dict[str, Any]], gold_nodes: List[Dict[str, Any]]) -> List[Tuple[int, int, str, float]]:
    candidates: List[Tuple[int, int, str, float]] = []
    for pi, p in enumerate(pred_nodes):
        for gi, g in enumerate(gold_nodes):
            if p["start"] == g["start"] and p["end"] == g["end"] and p["type"] == g["type"]:
                candidates.append((pi, gi, EXACT, 1.0))
                continue
            iou = overlap_iou(p["start"], p["end"], g["start"], g["end"])
            if p["type"] == g["type"] and iou >= SAME_TYPE_IOU_THRESHOLD:
                candidates.append((pi, gi, SAME_TYPE_OVERLAP, iou))
            elif iou >= ANY_TYPE_IOU_THRESHOLD:
                candidates.append((pi, gi, ANY_TYPE_OVERLAP, iou))
    priority = {EXACT: 3, SAME_TYPE_OVERLAP: 2, ANY_TYPE_OVERLAP: 1}
    candidates.sort(key=lambda x: (priority[x[2]], x[3]), reverse=True)
    return candidates


def greedy_match(pred_nodes: List[Dict[str, Any]], gold_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = generate_match_candidates(pred_nodes, gold_nodes)
    used_pred = set()
    used_gold = set()
    matched: List[Dict[str, Any]] = []
    for pi, gi, match_level, score in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        p = pred_nodes[pi]
        g = gold_nodes[gi]
        matched.append(
            {
                "pred_node_id": p["node_id"],
                "matched_gold_node_id": g["gold_node_id"],
                "match_level": match_level,
                "same_type": p["type"] == g["type"],
                "overlap_iou": score,
                "overlap_ratio": overlap_ratio(p["start"], p["end"], g["start"], g["end"]),
            }
        )
    for p in pred_nodes:
        if p["node_id"] not in used_pred:
            matched.append(
                {
                    "pred_node_id": p["node_id"],
                    "matched_gold_node_id": None,
                    "match_level": UNMATCHED,
                    "same_type": False,
                    "overlap_iou": 0.0,
                    "overlap_ratio": 0.0,
                }
            )
    matched.sort(key=lambda x: x["pred_node_id"])
    return matched


def build_eval_relation_matrices(
    pred_nodes: List[Dict[str, Any]],
    node_match: List[Dict[str, Any]],
    gold_relation_lookup: Dict[Tuple[int, int], str],
) -> List[List[int]]:
    n = len(pred_nodes)
    strict_exact = [[EVAL_RELATION_TO_ID["IGNORE"] for _ in range(n)] for _ in range(n)]
    exact_map: Dict[int, int] = {}
    for m in node_match:
        if m["matched_gold_node_id"] is None:
            continue
        pred_id = m["pred_node_id"]
        gold_id = m["matched_gold_node_id"]
        if m["match_level"] == EXACT:
            exact_map[pred_id] = gold_id
    for i in range(n):
        for j in range(n):
            if i == j:
                strict_exact[i][j] = EVAL_RELATION_TO_ID["IGNORE"]
                continue
            if i in exact_map and j in exact_map:
                rel = gold_relation_lookup.get((exact_map[i], exact_map[j]), "NONE")
                strict_exact[i][j] = EVAL_RELATION_TO_ID[rel]
    return strict_exact


def build_predicted_train_matrix(
    pred_nodes: List[Dict[str, Any]],
    node_match: List[Dict[str, Any]],
    gold_relation_lookup: Dict[Tuple[int, int], str],
    match_levels: set,
    unmatched_label: str = "NONE",
) -> List[List[int]]:
    n = len(pred_nodes)
    matrix = [[RELATION_TO_ID["NONE"] for _ in range(n)] for _ in range(n)]
    if unmatched_label not in {"NONE", "IGNORE"}:
        raise ValueError(f"unsupported unmatched_label: {unmatched_label}")
    pred_to_gold: Dict[int, int] = {}
    for m in node_match:
        if m["matched_gold_node_id"] is None:
            continue
        if m["match_level"] in match_levels:
            pred_to_gold[int(m["pred_node_id"])] = int(m["matched_gold_node_id"])

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = EVAL_RELATION_TO_ID["IGNORE"]
                continue
            if i in pred_to_gold and j in pred_to_gold:
                rel = gold_relation_lookup.get((pred_to_gold[i], pred_to_gold[j]), "NONE")
                matrix[i][j] = RELATION_TO_ID[rel]
            else:
                matrix[i][j] = (
                    EVAL_RELATION_TO_ID["IGNORE"]
                    if unmatched_label == "IGNORE"
                    else RELATION_TO_ID["NONE"]
                )
    return matrix


def build_fold_lookup(fold_json: Optional[Dict[str, Any]]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    if not fold_json:
        return lookup
    if "fold_assignment" in fold_json and isinstance(fold_json["fold_assignment"], dict):
        for doc_id, fold in fold_json["fold_assignment"].items():
            lookup[str(doc_id)] = int(fold)
    elif "fold_assignments" in fold_json and isinstance(fold_json["fold_assignments"], list):
        for item in fold_json["fold_assignments"]:
            doc_id = str(item.get("doc_id"))
            fold = item.get("fold")
            if fold is not None:
                lookup[doc_id] = int(fold)
    elif isinstance(fold_json, dict):
        for key, value in fold_json.items():
            if isinstance(value, int):
                lookup[str(key)] = value
    return lookup


def extract_fold_index(path: Path) -> Optional[int]:
    m = re.search(r"fold[_-]?(\d+)", str(path))
    return int(m.group(1)) if m else None


def extract_split_from_path(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def iter_prediction_root_candidates() -> List[Path]:
    outputs_dir = PROJECT_DIR / "outputs"
    candidates = [DEFAULT_PREDICTION_ROOT]
    explicit_names = ["event_extraction"]
    for name in explicit_names:
        path = outputs_dir / name
        if path not in candidates:
            candidates.append(path)
    if outputs_dir.exists():
        for path in sorted(outputs_dir.glob("event_extraction*")):
            if path not in candidates:
                candidates.append(path)
    return candidates


def resolve_prediction_root() -> Path:
    env_root = os.environ.get("PREDICTION_ROOT")
    if env_root:
        env_path = resolve_project_path(env_root)
        if env_path.exists():
            return env_path
        raise FileNotFoundError(
            f"PREDICTION_ROOT 已设置但路径不存在: {env_path}. "
            "请确认事件抽取输出目录是否正确。"
        )
    for path in iter_prediction_root_candidates():
        if not path.exists():
            continue
        for file_path in path.rglob("*.jsonl"):
            if looks_like_prediction_file(file_path):
                return path
    return DEFAULT_PREDICTION_ROOT


def looks_like_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ["prediction", "predictions", "event_predictions", "event_prediction"])


def is_valid_prediction_record(record: Dict[str, Any]) -> bool:
    if ALLOW_UNCHECKED_PREDICTIONS:
        return True
    return (
        record.get("prediction_schema") == "event_predictions"
        and record.get("task") == "document_level_event_span_extraction"
        and record.get("preferred_graph_input_field") == PREFERRED_NODE_SOURCE_FIELD
        and isinstance(record.get(PREFERRED_NODE_SOURCE_FIELD), list)
    )


def scan_prediction_files(root: Path) -> List[Path]:
    found: List[Path] = []
    if not root.exists():
        return found
    for path in root.rglob("*.jsonl"):
        if "inner_event_oof" in {part.lower() for part in path.parts}:
            continue
        if looks_like_prediction_file(path):
            found.append(path)
    found.sort(key=lambda p: str(p))
    return found


def infer_fold_and_split(record: Dict[str, Any], path: Path) -> Tuple[Optional[int], Optional[str]]:
    fold_index = record.get("fold_index")
    split = record.get("split")
    if fold_index is None:
        fold_index = extract_fold_index(path)
    if not split:
        split = extract_split_from_path(path)
    return (int(fold_index) if fold_index is not None else None, str(split) if split else None)


def build_predicted_sample(
    pred_record: Dict[str, Any],
    gold_doc: Dict[str, Any],
    fold_lookup: Dict[str, int],
    fold_index: int,
    split: str,
    source_path: Path,
    allowed_role_triples: set,
    role_constraint_source: str,
) -> Dict[str, Any]:
    text = normalize_text(gold_doc["text"])
    sent_spans = build_sentence_spans(text)
    para_spans = build_paragraph_spans(text)
    pred_nodes = normalize_predicted_nodes(pred_record)
    for node in pred_nodes:
        node["sent_id"] = find_span_bucket(node["start"], node["end"], sent_spans)
        node["para_id"] = find_span_bucket(node["start"], node["end"], para_spans)
    gold_nodes, event_id_to_node_id = build_gold_nodes(gold_doc)
    gold_relation_lookup = build_gold_relation_lookup(gold_doc, event_id_to_node_id)
    role_constraint_mask = build_role_constraint_mask(pred_nodes, allowed_role_triples=allowed_role_triples)
    node_match = greedy_match(pred_nodes, gold_nodes)
    strict_exact_eval_relation_matrix = build_eval_relation_matrices(
        pred_nodes=pred_nodes,
        node_match=node_match,
        gold_relation_lookup=gold_relation_lookup,
    )
    train_strict_exact_unmatched_none = build_predicted_train_matrix(
        pred_nodes=pred_nodes,
        node_match=node_match,
        gold_relation_lookup=gold_relation_lookup,
        match_levels={EXACT},
        unmatched_label="NONE",
    )
    train_strict_exact_unmatched_ignore = build_predicted_train_matrix(
        pred_nodes=pred_nodes,
        node_match=node_match,
        gold_relation_lookup=gold_relation_lookup,
        match_levels={EXACT},
        unmatched_label="IGNORE",
    )
    exact_match_count = sum(1 for m in node_match if m["match_level"] == EXACT)
    same_type_overlap_count = sum(1 for m in node_match if m["match_level"] in {EXACT, SAME_TYPE_OVERLAP})
    any_overlap_count = sum(1 for m in node_match if m["match_level"] in {EXACT, SAME_TYPE_OVERLAP, ANY_TYPE_OVERLAP})
    predicted_field = pred_record.get("preferred_graph_input_field") or PREFERRED_NODE_SOURCE_FIELD
    prediction_schema_name = pred_record.get("prediction_schema", "event_predictions")

    return {
        "doc_id": str(gold_doc["doc_id"]),
        "title": gold_doc.get("title", ""),
        "category": gold_doc.get("category", ""),
        "text": text,
        "fold_index": int(fold_index),
        "split": split,
        "schema": {
            "graph_sample_schema": GRAPH_SAMPLE_SCHEMA_NAME,
            "preferred_node_source_field": PREFERRED_NODE_SOURCE_FIELD,
            "event_types": EVENT_TYPES,
            "relation_types": RELATION_TYPES,
            "eval_relation_types": EVAL_RELATION_TYPES,
            "ignore_label": EVAL_RELATION_TO_ID["IGNORE"],
        },
        "nodes": pred_nodes,
        "edge_labels": [],
        "targets": {
            "train": None,
            "strict": strict_exact_eval_relation_matrix,
            "strict_exact": strict_exact_eval_relation_matrix,
            "train_strict_exact_unmatched_none": train_strict_exact_unmatched_none,
            "train_strict_exact_unmatched_ignore": train_strict_exact_unmatched_ignore,
        },
        "role_constraint_mask": role_constraint_mask,
        "gold_nodes_ref": gold_nodes,
        "gold_edge_labels_ref": [
            {"head": h, "tail": t, "type": rel_type}
            for (h, t), rel_type in sorted(gold_relation_lookup.items())
        ],
        "node_match": node_match,
        "relation_matrix": None,
        "strict_eval_relation_matrix": strict_exact_eval_relation_matrix,
        "strict_exact_eval_relation_matrix": strict_exact_eval_relation_matrix,
        "meta": {
            "sample_origin": "predicted",
            "prediction_source": str(source_path),
            "prediction_schema": prediction_schema_name,
            "preferred_graph_input_field": predicted_field,
            "role_constraint_source": role_constraint_source,
            "allowed_role_triple_count": len(allowed_role_triples),
            "pred_node_count": len(pred_nodes),
            "gold_node_count": len(gold_nodes),
            "gold_edge_count": len(gold_relation_lookup),
            "exact_match_count": exact_match_count,
            "same_type_overlap_match_count": same_type_overlap_count,
            "any_overlap_match_count": any_overlap_count,
            "fold_lookup_value": fold_lookup.get(str(gold_doc["doc_id"])),
            "node_source_distribution": dict(Counter(node.get("node_source_field", "unknown") for node in pred_nodes)),
        },
    }


def main() -> None:
    prediction_root = resolve_prediction_root()
    gold_docs = load_jsonl(GOLD_JSONL)
    gold_by_doc_id = {str(doc["doc_id"]): doc for doc in gold_docs}
    allowed_role_triples = build_default_role_triples()
    role_constraint_source = f"default_domain_rules_{ROLE_CONSTRAINT_PROFILE}"
    if INCLUDE_OBSERVED_ROLE_TRIPLES:
        allowed_role_triples = allowed_role_triples | build_observed_role_triples(gold_docs)
        role_constraint_source = f"default_domain_rules_{ROLE_CONSTRAINT_PROFILE}_union_observed_gold_type_pairs"
    fold_json = load_json(FOLD_JSON) if FOLD_JSON.exists() else None
    fold_lookup = build_fold_lookup(fold_json)
    prediction_files = scan_prediction_files(prediction_root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n================ RELATION GRAPH SAMPLE CONFIG ================")
    print(f"prediction_root: {prediction_root}")
    print(f"output_graph_sample_root: {OUTPUT_DIR}")
    print(f"summary_path: {SUMMARY_PATH}")
    print(f"run_folds_filter: {sorted(RUN_FOLDS_FILTER) if RUN_FOLDS_FILTER is not None else None}")
    print(f"include_splits_filter: {sorted(INCLUDE_SPLITS_FILTER) if INCLUDE_SPLITS_FILTER is not None else None}")
    print(f"role_constraint_profile: {ROLE_CONSTRAINT_PROFILE}")
    print(f"prediction_file_count: {len(prediction_files)}")
    print("===============================================================\n")

    summaries = []
    global_counter = Counter()
    global_node_source_counter = Counter()
    skipped_records = 0
    invalid_schema_records = 0
    duplicate_prediction_records: List[Dict[str, Any]] = []
    seen_prediction_keys = set()
    grouped_outputs: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_meta: Dict[Tuple[int, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "input_prediction_paths": set(),
            "pred_node_total": 0,
            "gold_node_total": 0,
            "exact_total": 0,
            "same_type_total": 0,
            "any_overlap_total": 0,
            "gold_edge_total": 0,
            "match_counter": Counter(),
            "node_source_counter": Counter(),
        }
    )

    for pred_path in prediction_files:
        pred_records = load_jsonl(pred_path)
        for record in pred_records:
            if not is_valid_prediction_record(record):
                invalid_schema_records += 1
                continue
            doc_id = str(record.get("doc_id", ""))
            if doc_id not in gold_by_doc_id:
                skipped_records += 1
                continue
            fold_index, split = infer_fold_and_split(record, pred_path)
            if fold_index is None:
                fold_index = fold_lookup.get(doc_id)
            if split is None:
                split = extract_split_from_path(pred_path) or "unknown"
            if fold_index is None:
                skipped_records += 1
                continue
            fold_index = int(fold_index)
            split = str(split).lower()
            if RUN_FOLDS_FILTER is not None and fold_index not in RUN_FOLDS_FILTER:
                skipped_records += 1
                continue
            if INCLUDE_SPLITS_FILTER is not None and split not in INCLUDE_SPLITS_FILTER:
                skipped_records += 1
                continue
            prediction_key = (fold_index, split, doc_id)
            if prediction_key in seen_prediction_keys:
                duplicate_prediction_records.append({
                    "fold_index": int(fold_index),
                    "split": str(split),
                    "doc_id": doc_id,
                    "path": str(pred_path),
                })
                continue
            seen_prediction_keys.add(prediction_key)

            sample = build_predicted_sample(
                pred_record=record,
                gold_doc=gold_by_doc_id[doc_id],
                fold_lookup=fold_lookup,
                fold_index=fold_index,
                split=split,
                source_path=pred_path,
                allowed_role_triples=allowed_role_triples,
                role_constraint_source=role_constraint_source,
            )
            key = (int(fold_index), str(split))
            grouped_outputs[key].append(sample)
            meta = grouped_meta[key]
            meta["input_prediction_paths"].add(str(pred_path))
            meta["pred_node_total"] += sample["meta"]["pred_node_count"]
            meta["gold_node_total"] += sample["meta"]["gold_node_count"]
            meta["exact_total"] += sample["meta"]["exact_match_count"]
            meta["same_type_total"] += sample["meta"]["same_type_overlap_match_count"]
            meta["any_overlap_total"] += sample["meta"]["any_overlap_match_count"]
            meta["gold_edge_total"] += sample["meta"]["gold_edge_count"]
            meta["node_source_counter"].update(sample["meta"]["node_source_distribution"])
            global_node_source_counter.update(sample["meta"]["node_source_distribution"])
            for m in sample["node_match"]:
                meta["match_counter"][m["match_level"]] += 1
                global_counter[m["match_level"]] += 1


    if duplicate_prediction_records:
        raise RuntimeError(
            "发现重复 prediction record，同一 (fold_index, split, doc_id) 只能出现一次: "
            + json.dumps(duplicate_prediction_records[:20], ensure_ascii=False)
        )
    if not grouped_outputs:
        raise RuntimeError(
            "没有可用的事件预测记录。请确认 PREDICTION_ROOT 指向事件抽取输出，且 prediction_schema/task/"
            "preferred_graph_input_field 与当前 schema 匹配。"
        )

    for (fold_index, split), built_samples in sorted(grouped_outputs.items()):
        out_path = OUTPUT_DIR / f"relation_graph_predicted_samples_fold_{fold_index}_{split}.jsonl"
        save_jsonl(built_samples, out_path)
        meta = grouped_meta[(fold_index, split)]
        pred_node_total = meta["pred_node_total"]
        gold_node_total = meta["gold_node_total"]
        exact_total = meta["exact_total"]
        same_type_total = meta["same_type_total"]
        any_overlap_total = meta["any_overlap_total"]
        summaries.append(
            {
                "fold_index": fold_index,
                "split": split,
                "input_prediction_paths": sorted(meta["input_prediction_paths"]),
                "output_sample_path": str(out_path),
                "document_count": len(built_samples),
                "avg_pred_node_count": pred_node_total / max(1, len(built_samples)),
                "avg_gold_node_count": gold_node_total / max(1, len(built_samples)),
                "exact_match_ratio": exact_total / max(1, pred_node_total),
                "same_type_overlap_match_ratio": same_type_total / max(1, pred_node_total),
                "any_overlap_match_ratio": any_overlap_total / max(1, pred_node_total),
                "gold_edge_total": meta["gold_edge_total"],
                "match_level_distribution": dict(meta["match_counter"]),
                "node_source_distribution": dict(meta["node_source_counter"]),
                "graph_sample_schema": GRAPH_SAMPLE_SCHEMA_NAME,
            }
        )

    overall = {
        "gold_path": str(GOLD_JSONL),
        "fold_path": str(FOLD_JSON),
        "prediction_root": str(prediction_root),
        "default_prediction_root": str(DEFAULT_PREDICTION_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "graph_sample_schema": GRAPH_SAMPLE_SCHEMA_NAME,
        "preferred_node_source_field": PREFERRED_NODE_SOURCE_FIELD,
        "role_constraint_source": role_constraint_source,
        "role_constraint_profile": ROLE_CONSTRAINT_PROFILE,
        "include_observed_role_triples": INCLUDE_OBSERVED_ROLE_TRIPLES,
        "allowed_role_triple_count": len(allowed_role_triples),
        "allowed_role_triples": sorted([list(item) for item in allowed_role_triples]),
        "prediction_file_count": len(prediction_files),
        "skipped_records": skipped_records,
        "invalid_schema_records": invalid_schema_records,
        "allow_unchecked_predictions": ALLOW_UNCHECKED_PREDICTIONS,
        "run_folds_filter": sorted(RUN_FOLDS_FILTER) if RUN_FOLDS_FILTER is not None else None,
        "include_splits_filter": sorted(INCLUDE_SPLITS_FILTER) if INCLUDE_SPLITS_FILTER is not None else None,
        "file_summaries": summaries,
        "overall_match_distribution": dict(global_counter),
        "overall_node_source_distribution": dict(global_node_source_counter),
        "prediction_root_candidates": [str(p) for p in iter_prediction_root_candidates()],
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
