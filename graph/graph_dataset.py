from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from graph.graph_schema import adapt_graph_sample, get_target_matrix
from graph.graph_core import EVENT_TYPE_TO_ID, RELATION_TYPES

DEFAULT_WINDOW_TOKEN_LENGTH = 448
DEFAULT_WINDOW_STRIDE = 192
TRAIN_TARGET_KEYS = {
    "train",
    "train_strict_exact_unmatched_none",
    "train_strict_exact_unmatched_ignore",
}


def load_fold_assignment(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    fold_assignment = payload.get("fold_assignment", payload)
    return {str(k): int(v) for k, v in fold_assignment.items()}


def split_samples_by_fold(samples: List[Dict], fold_assignment: Dict[str, int], test_fold: int, dev_fold: int):
    train_samples, dev_samples, test_samples = [], [], []
    for sample in samples:
        fold_idx = fold_assignment[str(sample["doc_id"])]
        if fold_idx == test_fold:
            test_samples.append(sample)
        elif fold_idx == dev_fold:
            dev_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, dev_samples, test_samples


def build_category_vocab(samples: List[Dict]) -> Dict[str, int]:
    categories = sorted({sample.get("category", "UNKNOWN") or "UNKNOWN" for sample in samples})
    return {name: idx for idx, name in enumerate(["UNKNOWN"] + [c for c in categories if c != "UNKNOWN"])}


def assert_fast_tokenizer(tokenizer) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("当前流程依赖 return_offsets_mapping，必须使用 fast tokenizer")


def count_labeled_features(features: List[Dict]) -> int:
    return sum(1 for feat in features if feat.get("target_matrix") is not None)


def find_token_start_index(offset_mapping: List[Tuple[int, int]], char_start: int) -> Optional[int]:
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        if start <= char_start < end:
            return idx
    return None


def find_token_end_index(offset_mapping: List[Tuple[int, int]], char_end: int) -> Optional[int]:
    target_char = char_end - 1
    if target_char < 0:
        return None
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        if start <= target_char < end:
            return idx
    return None


def build_windows(token_ids: List[int], offset_mapping: List[Tuple[int, int]], window_token_length: int, stride: int):
    windows = []
    total = len(token_ids)
    start = 0
    while start < total:
        end = min(total, start + window_token_length)
        windows.append({
            "token_start": start,
            "token_end": end,
            "token_ids": token_ids[start:end],
            "offset_mapping": offset_mapping[start:end],
        })
        if end >= total:
            break
        start += stride
    return windows


def build_window_encoding(tokenizer, token_ids_slice: List[int], offsets_slice: List[Tuple[int, int]]):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer 缺少 CLS 或 SEP token，无法构造窗口输入")

    input_ids = [cls_id] + token_ids_slice + [sep_id]
    attention_mask = [1] * len(input_ids)
    offset_mapping = [(0, 0)] + list(offsets_slice) + [(0, 0)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }


def build_document_feature(
    sample: Dict,
    tokenizer,
    category_to_id: Dict[str, int],
    target_key: str = "train",
    window_token_length: int = DEFAULT_WINDOW_TOKEN_LENGTH,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
) -> Dict:
    sample = adapt_graph_sample(sample)
    assert_fast_tokenizer(tokenizer)

    text = sample["text"]
    base_encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    full_token_ids = base_encoding["input_ids"]
    full_offsets = base_encoding["offset_mapping"]

    full_nodes = []
    for node in sample["nodes"]:
        token_start = find_token_start_index(full_offsets, int(node["start"]))
        token_end = find_token_end_index(full_offsets, int(node["end"]))
        if token_start is None or token_end is None or token_end < token_start:
            continue
        full_nodes.append({
            "node_id": int(node["node_id"]),
            "event_id": node.get("event_id", node.get("pred_event_id", f"P{int(node['node_id']) + 1}")),
            "start": int(node["start"]),
            "end": int(node["end"]),
            "text": node["text"],
            "type": node["type"],
            "sent_id": int(node.get("sent_id", 0)),
            "para_id": int(node.get("para_id", 0)),
            "score": node.get("score"),
            "support_count": node.get("support_count"),
            "view_consistency_score": node.get("view_consistency_score"),
            "support_views": node.get("support_views"),
            "node_quality": node.get("node_quality"),
            "token_start": token_start,
            "token_end": token_end,
        })

    windows = build_windows(
        token_ids=full_token_ids,
        offset_mapping=full_offsets,
        window_token_length=window_token_length,
        stride=window_stride,
    )

    window_features = []
    for window in windows:
        encoding = build_window_encoding(tokenizer, window["token_ids"], window["offset_mapping"])
        covered_node_ids = []
        local_node_token_spans = {}

        for node in full_nodes:
            if window["token_start"] <= node["token_start"] and node["token_end"] < window["token_end"]:
                local_start = node["token_start"] - window["token_start"] + 1
                local_end = node["token_end"] - window["token_start"] + 1
                covered_node_ids.append(node["node_id"])
                local_node_token_spans[node["node_id"]] = (local_start, local_end)

        window_features.append({
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "token_start": window["token_start"],
            "token_end": window["token_end"],
            "offset_mapping": encoding["offset_mapping"],
            "covered_node_ids": covered_node_ids,
            "local_node_token_spans": local_node_token_spans,
        })

    full_nodes = sorted(full_nodes, key=lambda x: x["node_id"])
    kept_old_node_ids = [int(node["node_id"]) for node in full_nodes]
    old_to_new = {}
    new_nodes = []
    for new_id, node in enumerate(full_nodes):
        old_id = int(node["node_id"])
        old_to_new[old_id] = new_id
        remapped = dict(node)
        remapped["orig_node_id"] = old_id
        remapped["node_id"] = new_id
        new_nodes.append(remapped)
    full_nodes = new_nodes

    for window in window_features:
        old_map = window.get("local_node_token_spans", {})
        new_map = {}
        for old_id, span in old_map.items():
            old_id = int(old_id)
            if old_id in old_to_new:
                new_map[old_to_new[old_id]] = span
        window["local_node_token_spans"] = new_map
        window["covered_node_ids"] = sorted(new_map.keys())

    raw_target_matrix = get_target_matrix(sample, target_key=target_key)
    raw_role_constraint_mask = sample["role_constraint_mask"]
    ignore_label = int(sample.get("ignore_label", -1))

    if len(kept_old_node_ids) == len(sample["nodes"]):
        target_matrix = None if raw_target_matrix is None else torch.tensor(raw_target_matrix, dtype=torch.long)
        role_constraint_mask = torch.tensor(raw_role_constraint_mask, dtype=torch.float)
    else:
        target_matrix = None
        if raw_target_matrix is not None:
            target_matrix = torch.tensor(
                [[raw_target_matrix[i][j] for j in kept_old_node_ids] for i in kept_old_node_ids],
                dtype=torch.long,
            )
        role_constraint_mask = torch.tensor(
            [
                [
                    [raw_role_constraint_mask[i][j][k] for k in range(len(RELATION_TYPES))]
                    for j in kept_old_node_ids
                ]
                for i in kept_old_node_ids
            ],
            dtype=torch.float,
        )

    if target_matrix is not None and target_key in TRAIN_TARGET_KEYS:
        none_id = RELATION_TYPES.index("NONE")
        node_count_for_mask = min(target_matrix.size(0), role_constraint_mask.size(0))
        for i in range(node_count_for_mask):
            for j in range(node_count_for_mask):
                label = int(target_matrix[i, j].item())
                if i == j or label < 0 or label == none_id:
                    continue
                if label < role_constraint_mask.size(-1):
                    role_constraint_mask[i, j, label] = 1.0

    node_count = len(full_nodes)
    node_type_ids = torch.tensor([EVENT_TYPE_TO_ID[node["type"]] for node in full_nodes], dtype=torch.long)
    sentence_ids = torch.tensor([min(node["sent_id"], 127) for node in full_nodes], dtype=torch.long)
    paragraph_ids = torch.tensor([min(node["para_id"], 63) for node in full_nodes], dtype=torch.long)
    node_order_ids = torch.arange(node_count, dtype=torch.long)
    category_name = sample.get("category", "UNKNOWN") or "UNKNOWN"
    category_id = category_to_id.get(category_name, category_to_id["UNKNOWN"])

    return {
        "doc_id": str(sample["doc_id"]),
        "title": sample.get("title", ""),
        "category": category_name,
        "category_id": torch.tensor(category_id, dtype=torch.long),
        "text": text,
        "nodes": full_nodes,
        "windows": window_features,
        "node_type_ids": node_type_ids,
        "sentence_ids": sentence_ids,
        "paragraph_ids": paragraph_ids,
        "node_order_ids": node_order_ids,
        "target_matrix": target_matrix,
        "role_constraint_mask": role_constraint_mask,
        "ignore_label": ignore_label,
        "target_key": target_key,
    }


class RelationGraphDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        category_to_id: Dict[str, int],
        target_key: str = "train",
        window_token_length: int = DEFAULT_WINDOW_TOKEN_LENGTH,
        window_stride: int = DEFAULT_WINDOW_STRIDE,
    ):
        self.target_key = target_key
        self.features = [
            build_document_feature(
                sample,
                tokenizer,
                category_to_id,
                target_key=target_key,
                window_token_length=window_token_length,
                window_stride=window_stride,
            )
            for sample in samples
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]


def collate_graph_batch(batch: List[Dict]) -> Dict:
    return {
        "doc_ids": [item["doc_id"] for item in batch],
        "titles": [item["title"] for item in batch],
        "categories": [item["category"] for item in batch],
        "category_ids": torch.stack([item["category_id"] for item in batch], dim=0),
        "texts": [item["text"] for item in batch],
        "nodes": [item["nodes"] for item in batch],
        "windows": [item["windows"] for item in batch],
        "node_type_ids": [item["node_type_ids"] for item in batch],
        "sentence_ids": [item["sentence_ids"] for item in batch],
        "paragraph_ids": [item["paragraph_ids"] for item in batch],
        "node_order_ids": [item["node_order_ids"] for item in batch],
        "target_matrices": [item["target_matrix"] for item in batch],
        "role_constraint_masks": [item["role_constraint_mask"] for item in batch],
        "ignore_labels": [item["ignore_label"] for item in batch],
        "target_keys": [item["target_key"] for item in batch],
    }
