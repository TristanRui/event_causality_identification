from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    candidates = [current.parent] + list(current.parent.parents)
    for p in candidates:
        if (p / "config.py").exists() and (p / "utils").exists() and (p / "graph").exists():
            return p
    return start.resolve().parents[1]


PROJECT_DIR = find_project_root(Path(__file__))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.io_utils import load_jsonl, save_json, save_jsonl
from utils.seed_utils import set_seed
from config import RANDOM_SEED
from graph.graph_schema import adapt_graph_sample
from graph.graph_dataset import (
    RelationGraphDataset,
    collate_graph_batch,
    build_document_feature,
)
from graph.graph_core import (
    RELATION_TYPES,
    RELATION_TYPE_TO_ID,
    RELATION_ID_TO_TYPE,
    RelationGraphModel,
    apply_hard_constraints_and_decode,
    compute_violation_count,
)
from graph.graph_metrics import (
    evaluate_decoded_docs,
    compute_end_to_end_graph_metrics,
    compute_graph_confidence,
)

INPUT_GRAPH_SAMPLE_ROOT = Path(
    os.environ.get(
        "INPUT_GRAPH_SAMPLE_ROOT",
        str(PROJECT_DIR / "data" / "processed" / "relation_graph_samples"),
    )
)
CHECKPOINT_NODE_SOURCE = os.environ.get("CHECKPOINT_NODE_SOURCE", os.environ.get("NODE_SOURCE", "predicted_events"))
CHECKPOINT_TRAIN_TARGET_KEY = os.environ.get(
    "CHECKPOINT_TRAIN_TARGET_KEY",
    os.environ.get("TRAIN_TARGET_KEY", "train_strict_exact_unmatched_ignore"),
)
DEFAULT_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "DEFAULT_CHECKPOINT_ROOT",
        str(PROJECT_DIR / "outputs" / "relation_model"),
    )
)
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", str(DEFAULT_CHECKPOINT_ROOT)))

OUTPUT_DIR = Path(
    os.environ.get(
        "DECODE_OUTPUT_DIR",
        str(PROJECT_DIR / "outputs" / "graph_predictions"),
    )
)
SUMMARY_PATH = OUTPUT_DIR / "decode_summary.json"

DECODE_SPLITS = [s.strip() for s in os.environ.get("DECODE_SPLITS", "test").split(",") if s.strip()]
BATCH_SIZE = int(os.environ.get("DECODE_BATCH_SIZE", "1"))
TOPK_RELATIONS_PER_DOC = int(os.environ.get("TOPK_RELATIONS_PER_DOC", "200"))
POSITIVE_MARGIN = float(os.environ.get("RELATION_POSITIVE_MARGIN", "0.5"))
DEFAULT_RELATION_THRESHOLD = float(os.environ.get("RELATION_SCORE_THRESHOLD", "0.0"))
PRETRAINED_MODEL_NAME = os.environ.get("PRETRAINED_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", "5"))
RUN_FOLDS = [
    int(item.strip())
    for item in os.environ.get("RUN_FOLDS", ",".join(str(i) for i in range(NUM_FOLDS))).split(",")
    if item.strip()
]

# The dataset target is used only to build the same cropped node/window view.
DATASET_TARGET_KEY = os.environ.get("DATASET_TARGET_KEY", "strict_exact")

_force_eval_target_keys_env = os.environ.get("FORCE_EVAL_TARGET_KEYS", "strict_exact").strip()
FORCE_EVAL_TARGET_KEYS: Optional[List[str]] = (
    [x.strip() for x in _force_eval_target_keys_env.split(",") if x.strip()]
    if _force_eval_target_keys_env
    else None
)

def parse_relation_thresholds() -> Dict[str, float]:
    raw = os.environ.get("RELATION_TYPE_THRESHOLDS", "").strip()
    thresholds = {name: DEFAULT_RELATION_THRESHOLD for name in RELATION_TYPES if name != "NONE"}
    if not raw:
        return thresholds
    for item in raw.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"RELATION_TYPE_THRESHOLDS 项格式错误: {item}")
        rel_type, value = item.split(":", 1)
        rel_type = rel_type.strip()
        if rel_type not in thresholds:
            raise ValueError(f"未知关系类型阈值: {rel_type}")
        thresholds[rel_type] = float(value)
    return thresholds


RELATION_TYPE_THRESHOLDS = parse_relation_thresholds()

FULL_PROCESS_COMPATIBLE_ROLE_TRIPLES = {
    ("ROOT_CAUSE", "FAILURE", "CAUSE"),
    ("ROOT_CAUSE", "PHENOMENON", "CAUSE"),
    ("FAILURE", "PHENOMENON", "CAUSE"),
    ("ACTION", "PHENOMENON", "TREAT"),
    ("ACTION", "FAILURE", "TREAT"),
    ("ACTION", "ROOT_CAUSE", "TREAT"),
    ("VERIFICATION", "ACTION", "VERIFY"),
    ("VERIFICATION", "PHENOMENON", "VERIFY"),
    ("VERIFICATION", "FAILURE", "VERIFY"),
    ("ACTION", "ACTION", "TEMPORAL"),
    ("VERIFICATION", "VERIFICATION", "TEMPORAL"),
}


def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    """Resolve a Hugging Face repo id to a local snapshot path before loading."""
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=True,
            local_dir=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法在本地 Hugging Face 缓存中解析模型 {model_name_or_path}。"
            "请先下载模型到本地缓存，或把 PRETRAINED_MODEL_NAME 设置为本地模型目录。"
        ) from exc


def resolve_eval_target_keys(samples: List[Dict[str, Any]]) -> List[str]:
    if FORCE_EVAL_TARGET_KEYS:
        return FORCE_EVAL_TARGET_KEYS
    if not samples:
        return ["strict_exact"]

    sample = adapt_graph_sample(samples[0])
    targets = sample.get("targets", {}) if isinstance(sample, dict) else {}
    keys: List[str] = []
    for key in ["strict_exact", "strict", "train"]:
        if targets.get(key) is not None:
            keys.append(key)
    return keys or ["strict_exact"]


def resolve_predicted_graph_sample_path(root: Path, fold_index: int, split_name: str) -> Optional[Path]:
    exact = root / f"relation_graph_predicted_samples_fold_{fold_index}_{split_name}.jsonl"
    if exact.exists():
        return exact

    patterns = [
        f"*fold_{fold_index}_{split_name}*.jsonl",
        f"*fold-{fold_index}-{split_name}*.jsonl",
        f"*{split_name}*fold_{fold_index}*.jsonl",
        f"*{split_name}*{fold_index}*.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def extract_positive_relations_from_matrix(
    pred_matrix: torch.Tensor,
    score_matrix: torch.Tensor,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    node_count = pred_matrix.size(0)
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            rel_id = int(pred_matrix[i, j].item())
            if rel_id <= 0:
                continue
            score = float(score_matrix[i, j].item())
            if score < threshold:
                continue
            edges.append(
                {
                    "head": int(i),
                    "tail": int(j),
                    "type": RELATION_ID_TO_TYPE[rel_id],
                    "score": score,
                }
            )
    edges.sort(key=lambda x: x["score"], reverse=True)
    return edges[:TOPK_RELATIONS_PER_DOC]


def apply_relation_thresholds(pred_matrix: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    filtered = pred_matrix.clone()
    none_id = RELATION_TYPE_TO_ID["NONE"]
    node_count = pred_matrix.size(0)
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            rel_id = int(pred_matrix[i, j].item())
            if rel_id <= 0:
                continue
            rel_type = RELATION_ID_TO_TYPE[rel_id]
            rel_prob = float(probs[i, j, rel_id].item())
            none_prob = float(probs[i, j, none_id].item())
            threshold = RELATION_TYPE_THRESHOLDS.get(rel_type, DEFAULT_RELATION_THRESHOLD)
            if rel_prob < threshold or (rel_prob - none_prob) < POSITIVE_MARGIN:
                filtered[i, j] = none_id
    return filtered


def build_model(category_count: int, pretrained_model_path: str):
    model = RelationGraphModel(
        pretrained_model_name=pretrained_model_path,
        category_count=category_count,
        dropout=DROPOUT,
    ).to(DEVICE)
    return model


def assert_fast_tokenizer(tokenizer) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("当前流程依赖 return_offsets_mapping，必须使用 fast tokenizer")


def validate_tokenizer_offset_mapping(tokenizer, sample_root: Path) -> None:
    sample_text = ""
    if sample_root.exists():
        for path in sorted(sample_root.glob("*.jsonl")):
            if "relation_graph_predicted_samples" not in path.name:
                continue
            rows = load_jsonl(str(path))
            for row in rows:
                sample_text = row.get("text", "")
                if sample_text:
                    break
            if sample_text:
                break
    if not sample_text:
        sample_text = "offset mapping smoke test"
    encoded = tokenizer(
        sample_text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping") or []
    if not any(int(s) < int(e) for s, e in offsets):
        raise RuntimeError("tokenizer 初始化成功但 offset_mapping 为空，无法解码关系图")


def load_category_vocab_from_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: Path) -> Dict[str, int]:
    category_to_id = checkpoint.get("category_to_id")
    if not isinstance(category_to_id, dict) or "UNKNOWN" not in category_to_id:
        raise RuntimeError(
            f"{checkpoint_path} is missing category_to_id. Re-train with scripts/05_train_relation_graph.py "
            "to keep category embeddings aligned between training and decoding."
        )
    return {str(k): int(v) for k, v in category_to_id.items()}


def build_aligned_target_lookup(
    samples: List[Dict[str, Any]],
    tokenizer,
    category_to_id: Dict[str, int],
    eval_target_keys: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build target matrices aligned with the dataset's cropped node view."""
    lookup: Dict[str, Dict[str, Any]] = {}

    all_target_keys: List[str] = []
    for key in [DATASET_TARGET_KEY] + list(eval_target_keys):
        if key not in all_target_keys:
            all_target_keys.append(key)

    for sample in samples:
        adapted = adapt_graph_sample(sample)
        doc_id = str(adapted["doc_id"])
        aligned_entry: Dict[str, Any] = {
            "doc_id": doc_id,
            "fold_index": adapted.get("fold_index"),
            "split": adapted.get("split"),
            "schema": adapted.get("schema", {}),
            "ignore_label": int(adapted.get("ignore_label", adapted.get("schema", {}).get("ignore_label", -1))),
            "node_match": adapted.get("node_match"),
            "gold_nodes_ref": adapted.get("gold_nodes_ref", []),
            "gold_edge_labels_ref": adapted.get("gold_edge_labels_ref", []),
            "meta": adapted.get("meta", {}),
            "targets": {},
        }

        dataset_feature = build_document_feature(
            adapted,
            tokenizer,
            category_to_id,
            target_key=DATASET_TARGET_KEY,
        )
        aligned_entry["nodes"] = dataset_feature["nodes"]

        for target_key in all_target_keys:
            feature = build_document_feature(
                adapted,
                tokenizer,
                category_to_id,
                target_key=target_key,
            )
            matrix = feature.get("target_matrix")
            aligned_entry["targets"][target_key] = matrix.tolist() if matrix is not None else None

        lookup[doc_id] = aligned_entry

    return lookup


def decode_batch(model, batch: Dict[str, Any], aligned_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    decoded_docs: List[Dict[str, Any]] = []
    for logits, role_mask, nodes, doc_id, title, category in zip(
        outputs["pair_logits"],
        batch["role_constraint_masks"],
        batch["nodes"],
        batch["doc_ids"],
        batch["titles"],
        batch["categories"],
    ):
        role_mask = role_mask.to(logits.device)
        masked_logits = logits.masked_fill(role_mask <= 0, -1e4)
        probs = torch.softmax(masked_logits, dim=-1)
        argmax_matrix = apply_hard_constraints_and_decode(logits, role_mask)
        argmax_scores = probs.gather(-1, argmax_matrix.unsqueeze(-1)).squeeze(-1)
        none_id = RELATION_TYPE_TO_ID["NONE"]
        none_scores = probs[..., none_id]
        positive_margins = argmax_scores - none_scores
        pred_matrix = apply_relation_thresholds(argmax_matrix, probs)
        pred_scores = probs.gather(-1, pred_matrix.unsqueeze(-1)).squeeze(-1)

        aligned_sample = aligned_lookup.get(str(doc_id), {})
        ignore_label = int(aligned_sample.get("ignore_label", aligned_sample.get("schema", {}).get("ignore_label", -1)))
        relations = extract_positive_relations_from_matrix(
            pred_matrix.detach().cpu(),
            pred_scores.detach().cpu(),
        )
        violation_count = compute_violation_count(pred_matrix.detach().cpu(), role_mask.detach().cpu())
        graph_confidence = compute_graph_confidence(
            pred_matrix.detach().cpu(),
            pred_scores.detach().cpu(),
            ignore_label=ignore_label,
        )

        decoded_docs.append(
            {
                "doc_id": str(doc_id),
                "title": title,
                "category": category,
                "fold_index": aligned_sample.get("fold_index"),
                "split": aligned_sample.get("split"),
                "schema": aligned_sample.get("schema", {"ignore_label": ignore_label}),
                "targets": aligned_sample.get("targets", {}),
                "node_match": aligned_sample.get("node_match"),
                "gold_nodes_ref": aligned_sample.get("gold_nodes_ref", []),
                "gold_edge_labels_ref": aligned_sample.get("gold_edge_labels_ref", []),
                "meta": aligned_sample.get("meta", {}),
                "nodes": [
                    {
                        "node_id": int(node["node_id"]),
                        "event_id": node.get("event_id", node.get("pred_event_id", f"P{int(node['node_id']) + 1}")),
                        "type": node["type"],
                        "text": node["text"],
                        "start": int(node["start"]),
                        "end": int(node["end"]),
                        "score": node.get("score"),
                        "support_count": node.get("support_count"),
                        "view_consistency_score": node.get("view_consistency_score"),
                        "support_views": node.get("support_views"),
                        "node_quality": node.get("node_quality"),
                    }
                    for node in nodes
                ],
                "argmax_relation_matrix": argmax_matrix.detach().cpu().tolist(),
                "argmax_score_matrix": argmax_scores.detach().cpu().tolist(),
                "none_score_matrix": none_scores.detach().cpu().tolist(),
                "argmax_positive_margin_matrix": positive_margins.detach().cpu().tolist(),
                "pred_relation_matrix": pred_matrix.detach().cpu().tolist(),
                "pred_score_matrix": pred_scores.detach().cpu().tolist(),
                "predicted_relations": relations,
                "graph_confidence": graph_confidence,
                "violation_count": violation_count,
            }
        )
    return decoded_docs


def summarize_decoded_docs(decoded_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    relation_type_counter = {rel_type: 0 for rel_type in RELATION_TYPES if rel_type != "NONE"}
    violation_total = 0
    process_violation_total = 0
    positive_relation_total = 0
    confidence_values: List[float] = []

    for doc in decoded_docs:
        violation_total += int(doc.get("violation_count", 0))
        if doc.get("graph_confidence") is not None:
            confidence_values.append(float(doc["graph_confidence"]))
        node_by_id = {int(node.get("node_id", idx)): node for idx, node in enumerate(doc.get("nodes", []) or [])}
        for rel in doc.get("predicted_relations", []):
            relation_type_counter[rel["type"]] += 1
            positive_relation_total += 1
            head = node_by_id.get(int(rel["head"]))
            tail = node_by_id.get(int(rel["tail"]))
            role_triple = (
                str(head.get("type", "")) if head else "",
                str(tail.get("type", "")) if tail else "",
                str(rel["type"]),
            )
            if role_triple not in FULL_PROCESS_COMPATIBLE_ROLE_TRIPLES:
                process_violation_total += 1

    return {
        "document_count": len(decoded_docs),
        "avg_graph_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "total_violation_count": violation_total,
        "process_constraint_violation_count": process_violation_total,
        "process_constraint_violation_rate": process_violation_total / max(1, positive_relation_total),
        "relation_type_distribution": relation_type_counter,
    }


def decode_split(
    model,
    sample_path: Path,
    tokenizer,
    category_to_id: Dict[str, int],
    split_name: str,
    fold_index: int,
    eval_target_keys: Optional[List[str]] = None,
):
    raw_samples = load_jsonl(str(sample_path))
    adapted_samples = [adapt_graph_sample(sample) for sample in raw_samples]
    if eval_target_keys is None:
        eval_target_keys = resolve_eval_target_keys(adapted_samples)

    dataset = RelationGraphDataset(
        adapted_samples,
        tokenizer,
        category_to_id,
        target_key=DATASET_TARGET_KEY,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_graph_batch)

    aligned_lookup = build_aligned_target_lookup(
        adapted_samples,
        tokenizer,
        category_to_id,
        eval_target_keys=eval_target_keys,
    )

    all_docs: List[Dict[str, Any]] = []
    for batch in tqdm(dataloader, desc=f"Decode fold {fold_index} {split_name}"):
        all_docs.extend(decode_batch(model, batch, aligned_lookup))

    split_dir = OUTPUT_DIR / f"fold_{fold_index}" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = split_dir / "graph_predictions.jsonl"
    save_jsonl(all_docs, str(prediction_path))

    summary: Dict[str, Any] = {
        "fold_index": fold_index,
        "split": split_name,
        "input_graph_sample_path": str(sample_path),
        "prediction_path": str(prediction_path),
        "dataset_target_key": DATASET_TARGET_KEY,
        "eval_target_keys": eval_target_keys,
        "relation_decode": {
            "positive_margin": POSITIVE_MARGIN,
            "default_relation_threshold": DEFAULT_RELATION_THRESHOLD,
            "relation_type_thresholds": RELATION_TYPE_THRESHOLDS,
        },
    }
    summary.update(summarize_decoded_docs(all_docs))

    per_target_metrics: Dict[str, Dict[str, Any]] = {}
    for target_key in eval_target_keys:
        per_target_metrics[target_key] = evaluate_decoded_docs(all_docs, target_key=target_key)
    summary["per_target_metrics"] = per_target_metrics
    for target_key in ["strict_exact"]:
        if target_key in per_target_metrics:
            summary[f"{target_key}_conditional_relation_f1"] = per_target_metrics[target_key]["f1"]

    end_to_end_metrics_by_target: Dict[str, Dict[str, Any]] = {}
    for target_key in ["strict_exact"]:
        end_to_end_metrics_by_target[target_key] = compute_end_to_end_graph_metrics(all_docs, target_key=target_key)
        summary[f"{target_key}_end_to_end_graph_f1"] = end_to_end_metrics_by_target[target_key]["f1"]
    summary["end_to_end_graph_metrics_by_target"] = end_to_end_metrics_by_target

    summary_path = split_dir / "decode_summary.json"
    save_json(summary, str(summary_path))
    return summary


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pretrained_model_path = resolve_local_pretrained_path(PRETRAINED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )

    assert_fast_tokenizer(tokenizer)
    validate_tokenizer_offset_mapping(tokenizer, INPUT_GRAPH_SAMPLE_ROOT)

    all_summaries: List[Dict[str, Any]] = []
    missing_sample_paths: List[str] = []

    print("\n================ DECODE CONFIG ================")
    print(f"INPUT_GRAPH_SAMPLE_ROOT: {INPUT_GRAPH_SAMPLE_ROOT}")
    print(f"CHECKPOINT_ROOT: {CHECKPOINT_ROOT}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"RUN_FOLDS: {RUN_FOLDS}")
    print(f"DECODE_SPLITS: {DECODE_SPLITS}")
    print(f"FORCE_EVAL_TARGET_KEYS: {FORCE_EVAL_TARGET_KEYS}")
    print("================================================\n")

    if NUM_FOLDS != 5 or RUN_FOLDS != [0, 1, 2, 3, 4]:
        raise ValueError(f"当前最终流程要求完整 5 折解码，NUM_FOLDS={NUM_FOLDS}, RUN_FOLDS={RUN_FOLDS}")

    for test_fold in RUN_FOLDS:
        checkpoint_path = CHECKPOINT_ROOT / f"fold_{test_fold}" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"缺少 fold {test_fold} checkpoint: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
        category_to_id = load_category_vocab_from_checkpoint(checkpoint, checkpoint_path)
        model = build_model(category_count=len(category_to_id), pretrained_model_path=pretrained_model_path)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        for split_name in DECODE_SPLITS:
            sample_path = resolve_predicted_graph_sample_path(INPUT_GRAPH_SAMPLE_ROOT, test_fold, split_name)
            if sample_path is None or not sample_path.exists():
                missing_sample_paths.append(f"fold={test_fold}, split={split_name}")
                raise FileNotFoundError(f"缺少 fold {test_fold} {split_name} 关系图样本文件")

            raw_samples = load_jsonl(str(sample_path))
            adapted_samples = [adapt_graph_sample(sample) for sample in raw_samples]
            eval_target_keys = resolve_eval_target_keys(adapted_samples)
            summary = decode_split(
                model=model,
                sample_path=sample_path,
                tokenizer=tokenizer,
                category_to_id=category_to_id,
                split_name=split_name,
                fold_index=test_fold,
                eval_target_keys=eval_target_keys,
            )
            all_summaries.append(summary)

    final_summary = {
        "input_graph_sample_root": str(INPUT_GRAPH_SAMPLE_ROOT),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "default_checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "checkpoint_node_source": CHECKPOINT_NODE_SOURCE,
        "checkpoint_train_target_key": CHECKPOINT_TRAIN_TARGET_KEY,
        "output_dir": str(OUTPUT_DIR),
        "num_folds": NUM_FOLDS,
        "run_folds": RUN_FOLDS,
        "decode_splits": DECODE_SPLITS,
        "dataset_target_key": DATASET_TARGET_KEY,
        "force_eval_target_keys": FORCE_EVAL_TARGET_KEYS,
        "relation_decode": {
            "positive_margin": POSITIVE_MARGIN,
            "default_relation_threshold": DEFAULT_RELATION_THRESHOLD,
            "relation_type_thresholds": RELATION_TYPE_THRESHOLDS,
        },
        "decode_summaries": all_summaries,
        "missing_sample_paths": missing_sample_paths,
    }
    save_json(final_summary, str(SUMMARY_PATH))
    print("\n================ PREDICTED GRAPH DECODE SUMMARY ================")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
