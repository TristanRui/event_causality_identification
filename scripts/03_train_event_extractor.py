import os
import sys
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Keep model loading offline and avoid background Hub calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import RANDOM_SEED
from utils.io_utils import load_jsonl, save_json, save_jsonl
from utils.seed_utils import set_seed

INPUT_SAMPLE_PATH = PROJECT_DIR / "data" / "processed" / "event_samples_all.jsonl"
INPUT_FOLD_PATH = PROJECT_DIR / "data" / "processed" / "event_document_folds.json"

RUN_DIR_NAME = os.environ.get("EVENT_RUN_DIR_NAME", "event_extraction")
OUTPUT_DIR = Path(os.environ.get("EVENT_OUTPUT_DIR", str(PROJECT_DIR / "outputs" / RUN_DIR_NAME)))
SUMMARY_PATH = OUTPUT_DIR / "cross_validation_summary.json"

PREDICTION_SCHEMA_NAME = "event_predictions"
PREDICTION_TASK_NAME = "document_level_event_span_extraction"
PREFERRED_GRAPH_INPUT_FIELD = "predicted_nodes"

PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
WINDOW_MAX_LENGTH = 448
WINDOW_STRIDE = 224
MAX_SPAN_LENGTH = 48
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
DROPOUT = 0.12
PATIENCE = 3
NUM_WORKERS = 4
AMP_ENABLED = torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATA_PARALLEL = False

PROPOSAL_PRE_NMS_TOPK = 224
FINAL_OVERLAP_THRESHOLD = 0.78
SPAN_LENGTH_PENALTY = 0.006

PROPOSAL_SCORE_THRESHOLD_BY_TYPE = {
    "PHENOMENON": 0.12,
    "FAILURE": 0.12,
    "ROOT_CAUSE": 0.10,
    "ACTION": 0.16,
    "VERIFICATION": 0.14,
}
FINAL_SCORE_THRESHOLD_BY_TYPE = {
    "PHENOMENON": 0.12,
    "FAILURE": 0.11,
    "ROOT_CAUSE": 0.08,
    "ACTION": 0.14,
    "VERIFICATION": 0.13,
}
MAX_SPAN_LENGTH_BY_TYPE = {
    "PHENOMENON": 56,
    "FAILURE": 48,
    "ROOT_CAUSE": 72,
    "ACTION": 48,
    "VERIFICATION": 48,
}

CANDIDATE_LIMIT_CONFIG = {
    "PHENOMENON": 5,
    "FAILURE": 5,
    "ROOT_CAUSE": 7,
    "ACTION": 6,
    "VERIFICATION": 5,
}

SPAN_QUALITY_LOSS_WEIGHT = 0.5
NEGATIVE_SPANS_PER_POSITIVE = 3
MIN_NEGATIVE_SPANS = 6
MAX_NEGATIVE_SPANS = 24
TYPE_EMBED_DIM = 16

EVENT_TYPES = [
    "PHENOMENON",
    "FAILURE",
    "ROOT_CAUSE",
    "ACTION",
    "VERIFICATION",
]
EVENT_TYPE_TO_ID = {name: idx for idx, name in enumerate(EVENT_TYPES)}
EVENT_ID_TO_TYPE = {idx: name for name, idx in EVENT_TYPE_TO_ID.items()}

VIEW_TYPES = ["document", "section", "bridge", "unknown"]
VIEW_TYPE_TO_ID = {name: idx for idx, name in enumerate(VIEW_TYPES)}
SECTION_KEYS = ["unknown", "document", "phenomenon", "cause", "process", "loss"]
SECTION_KEY_TO_ID = {name: idx for idx, name in enumerate(SECTION_KEYS)}

SECTION_EVENT_PRIOR = {
    "phenomenon": {"PHENOMENON": 0.06, "FAILURE": 0.03, "ROOT_CAUSE": -0.01, "ACTION": -0.02, "VERIFICATION": -0.02},
    "cause": {"PHENOMENON": -0.01, "FAILURE": 0.02, "ROOT_CAUSE": 0.08, "ACTION": -0.01, "VERIFICATION": -0.02},
    "process": {"PHENOMENON": -0.01, "FAILURE": 0.00, "ROOT_CAUSE": -0.02, "ACTION": 0.08, "VERIFICATION": 0.06},
    "loss": {"PHENOMENON": 0.03, "FAILURE": 0.05, "ROOT_CAUSE": -0.02, "ACTION": -0.02, "VERIFICATION": -0.01},
    "document": {"PHENOMENON": 0.0, "FAILURE": 0.0, "ROOT_CAUSE": 0.0, "ACTION": 0.0, "VERIFICATION": 0.0},
    "unknown": {"PHENOMENON": 0.0, "FAILURE": 0.0, "ROOT_CAUSE": 0.0, "ACTION": 0.0, "VERIFICATION": 0.0},
}

def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    """Resolve a Hugging Face repo id to a local snapshot before loading."""
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)

    try:
        local_dir = snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=True,
            local_dir=None,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"无法在本地 Hugging Face 缓存中解析模型 {model_name_or_path}。"
            "请先确保该模型已经被下载到本地缓存，或者把 PRETRAINED_MODEL_NAME 改成一个本地目录。"
        ) from e

    return local_dir


def assert_fast_tokenizer(tokenizer) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("当前流程依赖 return_offsets_mapping，必须使用 fast tokenizer")


def validate_tokenizer_offset_mapping(tokenizer, samples: List[Dict]) -> None:
    sample_text = ""
    for sample in samples:
        sample_text = sample.get("text", "") or sample.get("raw_text", "")
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
        raise RuntimeError("tokenizer 初始化成功但 offset_mapping 为空，无法进行字符位置到 token 位置映射")



def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_fold_assignment(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_samples_by_fold(samples: List[Dict], fold_assignment: Dict[str, int], test_fold: int, dev_fold: int):
    train_samples, dev_samples, test_samples = [], [], []
    for sample in samples:
        fold_index = fold_assignment[str(sample["doc_id"])]
        if fold_index == test_fold:
            test_samples.append(sample)
        elif fold_index == dev_fold:
            dev_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, dev_samples, test_samples


def build_document_gold_spans(event_mentions: List[Dict]) -> List[Tuple[int, int, str]]:
    spans = []
    for event in event_mentions:
        event_type = event.get("type", "")
        start = int(event.get("start", -1))
        end = int(event.get("end", -1))
        if event_type not in EVENT_TYPE_TO_ID:
            continue
        if start < 0 or end <= start:
            continue
        spans.append((start, end, event_type))
    return spans


def tokenize_document_without_truncation(text: str, tokenizer):
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    return {
        "input_ids": encoding["input_ids"],
        "offset_mapping": encoding["offset_mapping"],
        "text": text,
    }


def find_overlapping_token_start_index(offset_mapping: List[Tuple[int, int]], char_start: int):
    for idx, (s, e) in enumerate(offset_mapping):
        if s == e:
            continue
        if e > char_start:
            return idx
    return None


def find_overlapping_token_end_index(offset_mapping: List[Tuple[int, int]], char_end: int):
    for idx in range(len(offset_mapping) - 1, -1, -1):
        s, e = offset_mapping[idx]
        if s == e:
            continue
        if s < char_end:
            return idx
    return None


def resolve_token_span(offset_mapping: List[Tuple[int, int]], char_start: int, char_end: int):
    token_start = find_token_start_index(offset_mapping, char_start)
    if token_start is None:
        token_start = find_overlapping_token_start_index(offset_mapping, char_start)
    token_end = find_token_end_index(offset_mapping, char_end)
    if token_end is None:
        token_end = find_overlapping_token_end_index(offset_mapping, char_end)
    if token_start is None or token_end is None or token_end < token_start:
        return None, None
    return token_start, token_end


def infer_primary_section_key(view: Dict) -> str:
    if view.get("view_type") == "document":
        return "document"
    source_sections = view.get("source_sections", [])
    if source_sections:
        section_key = source_sections[0]
        if section_key in SECTION_KEY_TO_ID:
            return section_key
    return "unknown"


def build_local_windows(global_start: int, global_end: int):
    windows = []
    if global_end <= global_start:
        return windows
    start = global_start
    content_length = WINDOW_MAX_LENGTH - 2
    while start < global_end:
        end = min(global_end, start + content_length)
        windows.append((start, end))
        if end >= global_end:
            break
        next_start = end - WINDOW_STRIDE
        if next_start <= start:
            next_start = end
        start = next_start
    return windows


def build_view_guided_windows(document: Dict, token_ids: List[int], offsets: List[Tuple[int, int]]):
    views = document.get("views", []) or [{
        "view_id": "document",
        "view_type": "document",
        "source_sections": ["document"],
        "start": 0,
        "end": len(document.get("text", "")),
        "text": document.get("text", ""),
    }]
    windows = []
    seen = set()
    for view in views:
        char_start = int(view.get("start", 0))
        char_end = int(view.get("end", len(document.get("text", ""))))
        token_start, token_end = resolve_token_span(offsets, char_start, char_end)
        if token_start is None or token_end is None:
            continue
        token_end = token_end + 1
        view_type = view.get("view_type", "unknown")
        primary_section_key = infer_primary_section_key(view)
        for local_start, local_end in build_local_windows(token_start, token_end):
            key = (local_start, local_end, view.get("view_id", ""))
            if key in seen:
                continue
            seen.add(key)
            windows.append({
                "token_start": local_start,
                "token_end": local_end,
                "view_id": view.get("view_id", ""),
                "view_type": view_type if view_type in VIEW_TYPE_TO_ID else "unknown",
                "primary_section_key": primary_section_key,
                "source_sections": view.get("source_sections", []),
            })
    if not windows:
        for local_start, local_end in build_local_windows(0, len(token_ids)):
            windows.append({
                "token_start": local_start,
                "token_end": local_end,
                "view_id": "document",
                "view_type": "document",
                "primary_section_key": "document",
                "source_sections": ["document"],
            })
    return windows


def section_prior_bonus(event_type: str, section_key: str) -> float:
    return SECTION_EVENT_PRIOR.get(section_key, SECTION_EVENT_PRIOR["unknown"]).get(event_type, 0.0)


def build_sliding_windows(token_ids: List[int]):
    if not token_ids:
        return [(0, 0)]
    windows = []
    start = 0
    content_length = WINDOW_MAX_LENGTH - 2  # CLS/SEP
    while start < len(token_ids):
        end = min(len(token_ids), start + content_length)
        windows.append((start, end))
        if end >= len(token_ids):
            break
        next_start = end - WINDOW_STRIDE
        if next_start <= start:
            next_start = end
        start = next_start
    return windows


def find_token_start_index(offset_mapping: List[Tuple[int, int]], char_start: int):
    for idx, (s, e) in enumerate(offset_mapping):
        if s == e:
            continue
        if s <= char_start < e:
            return idx
    return None


def find_token_end_index(offset_mapping: List[Tuple[int, int]], char_end: int):
    target = char_end - 1
    if target < 0:
        return None
    for idx, (s, e) in enumerate(offset_mapping):
        if s == e:
            continue
        if s <= target < e:
            return idx
    return None


def build_window_encoding(tokenizer, token_ids_slice: List[int], offsets_slice: List[Tuple[int, int]]):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("tokenizer 缺少 cls_token_id 或 sep_token_id")
    input_ids = [cls_id] + token_ids_slice + [sep_id]
    attention_mask = [1] * len(input_ids)
    offset_mapping = [(0, 0)] + offsets_slice + [(0, 0)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }


def project_gold_spans_to_window(gold_spans, window_offset_mapping):
    num_types = len(EVENT_TYPES)
    seq_len = len(window_offset_mapping)
    start_labels = torch.zeros(seq_len, num_types, dtype=torch.float)
    end_labels = torch.zeros(seq_len, num_types, dtype=torch.float)
    valid_mask = torch.tensor([1.0 if s < e else 0.0 for s, e in window_offset_mapping], dtype=torch.float)
    window_gold_char_spans = []
    window_gold_token_spans = []
    valid_offsets = [(s, e) for s, e in window_offset_mapping if s < e]
    if not valid_offsets:
        return start_labels, end_labels, valid_mask, window_gold_char_spans, window_gold_token_spans

    window_char_start = min(s for s, _ in valid_offsets)
    window_char_end = max(e for _, e in valid_offsets)

    for char_start, char_end, event_type in gold_spans:
        if char_start < window_char_start or char_end > window_char_end:
            continue
        token_start, token_end = resolve_token_span(window_offset_mapping, char_start, char_end)
        if token_start is None or token_end is None or token_end < token_start:
            continue
        type_id = EVENT_TYPE_TO_ID[event_type]
        start_labels[token_start, type_id] = 1.0
        end_labels[token_end, type_id] = 1.0
        window_gold_char_spans.append((char_start, char_end, event_type))
        window_gold_token_spans.append((token_start, token_end + 1, event_type))

    return start_labels, end_labels, valid_mask, window_gold_char_spans, window_gold_token_spans


def sample_span_training_examples(valid_mask, gold_token_spans):
    valid_positions = [i for i, v in enumerate(valid_mask.tolist()) if v > 0]
    if not valid_positions:
        return []

    gold_set = {(s, e, t) for s, e, t in gold_token_spans}
    examples = []
    rng = random.Random(42 + len(gold_token_spans) + sum(valid_positions))

    for s, e, t in gold_token_spans:
        examples.append((s, e, EVENT_TYPE_TO_ID[t], 1))

    negative_set = set()
    max_global_span = max(MAX_SPAN_LENGTH_BY_TYPE.values())

    def try_add_negative(s, e, event_type):
        if e <= s or e - s > max_global_span:
            return
        if s not in valid_positions or (e - 1) not in valid_positions:
            return
        if (s, e, event_type) in gold_set:
            return
        negative_set.add((s, e, event_type))

    # Hard negatives from boundary perturbations.
    for s, e, t in gold_token_spans:
        for delta_left in (-3, -2, -1, 1, 2, 3):
            try_add_negative(s + delta_left, e, t)
        for delta_right in (-3, -2, -1, 1, 2, 3):
            try_add_negative(s, e + delta_right, t)
        for delta_left in (-2, -1, 1, 2):
            for delta_right in (-2, -1, 1, 2):
                try_add_negative(s + delta_left, e + delta_right, t)
        if e - s > 2:
            try_add_negative(s + 1, e - 1, t)
        try_add_negative(max(valid_positions[0], s - 1), min(valid_positions[-1] + 1, e + 1), t)

        # Same-span negatives with commonly confused event types.
        confusion_types = {
            "PHENOMENON": ["FAILURE", "ROOT_CAUSE"],
            "FAILURE": ["PHENOMENON", "ROOT_CAUSE"],
            "ROOT_CAUSE": ["FAILURE", "PHENOMENON"],
            "ACTION": ["VERIFICATION"],
            "VERIFICATION": ["ACTION"],
        }
        for other_t in confusion_types.get(t, []):
            try_add_negative(s, e, other_t)

    # Cross-span negatives built from neighboring gold spans.
    sorted_golds = sorted(gold_token_spans, key=lambda x: (x[0], x[1]))
    for idx in range(len(sorted_golds) - 1):
        s1, e1, t1 = sorted_golds[idx]
        s2, e2, t2 = sorted_golds[idx + 1]
        try_add_negative(s1, e2, t1)
        try_add_negative(s1, e2, t2)
        mid_start = min(e1, s2)
        mid_end = max(e1, s2)
        if mid_end > mid_start:
            try_add_negative(mid_start, mid_end, t1)
            try_add_negative(mid_start, mid_end, t2)

    # High-overlap random negatives around gold spans.
    for s, e, t in gold_token_spans:
        for _ in range(4):
            left = max(valid_positions[0], s + rng.choice([-2, -1, 0, 1, 2]))
            right = min(valid_positions[-1] + 1, e + rng.choice([-2, -1, 0, 1, 2]))
            if right > left:
                try_add_negative(left, right, t)

    # Fill the remaining quota with random negatives.
    target_negative_count = min(MAX_NEGATIVE_SPANS, max(MIN_NEGATIVE_SPANS, len(gold_token_spans) * NEGATIVE_SPANS_PER_POSITIVE + 8))
    attempts = 0
    while len(negative_set) < target_negative_count and attempts < target_negative_count * 30:
        attempts += 1
        s = rng.choice(valid_positions)
        event_type = rng.choice(EVENT_TYPES)
        max_span = MAX_SPAN_LENGTH_BY_TYPE[event_type]
        max_end = min(valid_positions[-1] + 1, s + max_span)
        if max_end <= s + 1:
            continue
        e = rng.randint(s + 1, max_end)
        try_add_negative(s, e, event_type)

    for s, e, t in sorted(negative_set):
        examples.append((s, e, EVENT_TYPE_TO_ID[t], 0))
    return examples


def build_windowed_training_targets(document: Dict, tokenizer):
    text = document.get("text", "")
    doc_id = str(document.get("doc_id", ""))
    gold_spans = build_document_gold_spans(document.get("event_mentions", []))
    sections = document.get("sections", [])
    views = document.get("views", [])

    full = tokenize_document_without_truncation(text, tokenizer)
    token_ids = full["input_ids"]
    offsets = full["offset_mapping"]
    windows = build_view_guided_windows(document, token_ids, offsets)

    features = []
    for window_id, window in enumerate(windows):
        start_token = window["token_start"]
        end_token = window["token_end"]
        token_slice = token_ids[start_token:end_token]
        offset_slice = offsets[start_token:end_token]
        encoding = build_window_encoding(tokenizer, token_slice, offset_slice)
        start_labels, end_labels, valid_mask, window_gold_char_spans, window_gold_token_spans = project_gold_spans_to_window(gold_spans, encoding["offset_mapping"])
        span_training_examples = sample_span_training_examples(valid_mask, window_gold_token_spans)

        features.append({
            "doc_id": doc_id,
            "window_id": window_id,
            "text": text,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": encoding["offset_mapping"],
            "start_labels": start_labels,
            "end_labels": end_labels,
            "valid_mask": valid_mask,
            "window_gold_spans": window_gold_char_spans,
            "window_gold_token_spans": window_gold_token_spans,
            "span_training_examples": span_training_examples,
            "document_gold_spans": gold_spans,
            "sections": sections,
            "views": views,
            "view_id": window["view_id"],
            "view_type": window["view_type"],
            "view_type_id": VIEW_TYPE_TO_ID[window["view_type"]],
            "primary_section_key": window["primary_section_key"],
            "section_key_id": SECTION_KEY_TO_ID.get(window["primary_section_key"], SECTION_KEY_TO_ID["unknown"]),
            "source_sections": window.get("source_sections", []),
        })

    return features


class DocumentWindowDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer):
        self.features = []
        self.document_count = len(items)
        for item in items:
            self.features.extend(build_windowed_training_targets(item, tokenizer))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            "input_ids": torch.tensor(feat["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.long),
            "valid_mask": feat["valid_mask"],
            "start_labels": feat["start_labels"],
            "end_labels": feat["end_labels"],
            "offset_mapping": feat["offset_mapping"],
            "window_gold_spans": feat["window_gold_spans"],
            "window_gold_token_spans": feat["window_gold_token_spans"],
            "span_training_examples": feat["span_training_examples"],
            "document_gold_spans": feat["document_gold_spans"],
            "text": feat["text"],
            "doc_id": feat["doc_id"],
            "window_id": feat["window_id"],
            "sections": feat["sections"],
            "views": feat["views"],
            "view_id": feat["view_id"],
            "view_type": feat["view_type"],
            "view_type_id": feat["view_type_id"],
            "primary_section_key": feat["primary_section_key"],
            "section_key_id": feat["section_key_id"],
            "source_sections": feat["source_sections"],
        }


def collate_batch(batch):
    input_ids = nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
    valid_mask = nn.utils.rnn.pad_sequence([x["valid_mask"] for x in batch], batch_first=True, padding_value=0.0)
    start_labels = nn.utils.rnn.pad_sequence([x["start_labels"] for x in batch], batch_first=True, padding_value=0.0)
    end_labels = nn.utils.rnn.pad_sequence([x["end_labels"] for x in batch], batch_first=True, padding_value=0.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "valid_mask": valid_mask,
        "start_labels": start_labels,
        "end_labels": end_labels,
        "offset_mappings": [x["offset_mapping"] for x in batch],
        "window_gold_spans": [x["window_gold_spans"] for x in batch],
        "window_gold_token_spans": [x["window_gold_token_spans"] for x in batch],
        "span_training_examples": [x["span_training_examples"] for x in batch],
        "document_gold_spans": [x["document_gold_spans"] for x in batch],
        "texts": [x["text"] for x in batch],
        "doc_ids": [x["doc_id"] for x in batch],
        "window_ids": [x["window_id"] for x in batch],
        "sections": [x["sections"] for x in batch],
        "views": [x["views"] for x in batch],
        "view_ids": [x["view_id"] for x in batch],
        "view_types": [x["view_type"] for x in batch],
        "view_type_ids": torch.tensor([x["view_type_id"] for x in batch], dtype=torch.long),
        "primary_section_keys": [x["primary_section_key"] for x in batch],
        "section_key_ids": torch.tensor([x["section_key_id"] for x in batch], dtype=torch.long),
        "source_sections": [x["source_sections"] for x in batch],
    }


class MiddleStableSpanModel(nn.Module):
    def __init__(self, pretrained_model_name: str, num_types: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, local_files_only=True, use_safetensors=False, trust_remote_code=False)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.start_classifier = nn.Linear(hidden, num_types)
        self.end_classifier = nn.Linear(hidden, num_types)
        self.type_embedding = nn.Embedding(num_types, TYPE_EMBED_DIM)
        self.view_type_embedding = nn.Embedding(len(VIEW_TYPES), 8)
        self.section_embedding = nn.Embedding(len(SECTION_KEYS), 8)
        self.span_mlp = nn.Sequential(
            nn.Linear(hidden * 3 + TYPE_EMBED_DIM + 8 + 8, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(outputs.last_hidden_state)
        return {
            "sequence_output": seq,
            "start_logits": self.start_classifier(seq),
            "end_logits": self.end_classifier(seq),
        }

    def score_spans(self, sequence_output: torch.Tensor, span_examples: List[Tuple[int, int, int]], view_type_id: int = 0, section_key_id: int = 0):
        if not span_examples:
            return torch.empty(0, device=sequence_output.device)
        reps = []
        view_type_vec = self.view_type_embedding.weight[int(view_type_id)]
        section_vec = self.section_embedding.weight[int(section_key_id)]
        for s, e, type_id in span_examples:
            start_vec = sequence_output[s]
            end_vec = sequence_output[e - 1]
            mean_vec = sequence_output[s:e].mean(dim=0)
            type_vec = self.type_embedding.weight[type_id]
            rep = torch.cat([start_vec, end_vec, mean_vec, type_vec, view_type_vec, section_vec], dim=-1)
            reps.append(rep)
        rep_tensor = torch.stack(reps, dim=0)
        return self.span_mlp(rep_tensor).squeeze(-1)


def build_positive_class_weights(dataset: DocumentWindowDataset):
    num_types = len(EVENT_TYPES)
    positive_count = torch.zeros(num_types, dtype=torch.float)
    valid_count = 0.0
    for feat in dataset.features:
        valid_count += feat["valid_mask"].sum().item()
        positive_count += feat["start_labels"].sum(dim=0)
        positive_count += feat["end_labels"].sum(dim=0)
    total_slots = valid_count * 2.0
    negative_count = total_slots - positive_count
    positive_weight = negative_count / positive_count.clamp(min=1.0)
    positive_weight = torch.clamp(positive_weight, min=5.0, max=100.0)
    return positive_weight


def build_span_quality_pos_weight(dataset: DocumentWindowDataset):
    positive = 0.0
    negative = 0.0
    for feat in dataset.features:
        for _, _, _, label in feat["span_training_examples"]:
            if label == 1:
                positive += 1
            else:
                negative += 1
    if positive <= 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, min(20.0, negative / positive)), dtype=torch.float)


def compute_boundary_loss(start_logits, end_logits, start_labels, end_labels, valid_mask, positive_weight):
    loss_fn = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=positive_weight.to(start_logits.device).view(1, 1, -1),
    )
    start_loss = loss_fn(start_logits, start_labels)
    end_loss = loss_fn(end_logits, end_labels)
    mask = valid_mask.unsqueeze(-1)
    denom = (mask.sum() * start_logits.size(-1)).clamp(min=1.0)
    return ((start_loss + end_loss) * mask).sum() / (2.0 * denom)


def compute_span_quality_loss(model, sequence_output, span_training_examples_batch, view_type_ids, section_key_ids, pos_weight):
    losses = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight.to(sequence_output.device))
    for batch_idx, examples in enumerate(span_training_examples_batch):
        if not examples:
            continue
        span_inputs = [(s, e, t) for s, e, t, _ in examples]
        labels = torch.tensor([label for _, _, _, label in examples], dtype=torch.float, device=sequence_output.device)
        logits = model.score_spans(
            sequence_output[batch_idx],
            span_inputs,
            int(view_type_ids[batch_idx].item()),
            int(section_key_ids[batch_idx].item()),
        )
        if logits.numel() == 0:
            continue
        losses.append(loss_fn(logits, labels))
    if not losses:
        return torch.tensor(0.0, device=sequence_output.device)
    return torch.stack(losses).mean()


def normalize_span_text(text: str) -> str:
    text = text.strip()
    text = text.strip("：:；;，,。.!！?？、 ")
    return text


def is_heading_term_only(text: str) -> bool:
    heading_terms = {
        "分解", "复装", "验证", "检查", "换件",
        "原因分析", "异常现象描述", "排查与处理过程",
    }
    return normalize_span_text(text) in heading_terms


def overlap_ratio_on_shorter_span(span_a, span_b) -> float:
    start_a, end_a, _ = span_a[:3]
    start_b, end_b, _ = span_b[:3]
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    if overlap <= 0:
        return 0.0
    len_a = max(1, end_a - start_a)
    len_b = max(1, end_b - start_b)
    return overlap / min(len_a, len_b)


def convert_token_span_to_char_span(text: str, offset_mapping: List[Tuple[int, int]], token_span: Tuple[int, int, str]):
    token_start, token_end, event_type = token_span
    if token_start < 0 or token_end <= token_start or token_end - 1 >= len(offset_mapping):
        return {"type": event_type, "text": "", "char_start": -1, "char_end": -1}
    char_start = offset_mapping[token_start][0]
    char_end = offset_mapping[token_end - 1][1]
    return {
        "type": event_type,
        "text": text[char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
    }


def convert_token_span_tuple_to_char_span(text: str, offset_mapping: List[Tuple[int, int]], token_span: Tuple[int, int, str]):
    info = convert_token_span_to_char_span(text, offset_mapping, token_span)
    return (info["char_start"], info["char_end"], info["type"])


def build_span_candidates(start_probabilities, end_probabilities, valid_positions: List[int]):
    candidates = []
    valid_positions = sorted(int(pos) for pos in valid_positions)
    if not valid_positions:
        return candidates

    def pick_positions(prob_vector, threshold: float, fallback_topk: int = 12, hard_topk: int = 24):
        positions = [i for i in valid_positions if float(prob_vector[i]) >= threshold]
        if positions:
            positions = sorted(positions, key=lambda i: float(prob_vector[i]), reverse=True)[:hard_topk]
            return positions

        ranked = sorted(valid_positions, key=lambda i: float(prob_vector[i]), reverse=True)[:fallback_topk]
        ranked = [i for i in ranked if float(prob_vector[i]) > 0.02]
        return ranked

    for type_id in range(len(EVENT_TYPES)):
        event_type = EVENT_ID_TO_TYPE[type_id]
        proposal_threshold = PROPOSAL_SCORE_THRESHOLD_BY_TYPE[event_type]
        max_span_length = MAX_SPAN_LENGTH_BY_TYPE[event_type]

        start_positions = pick_positions(start_probabilities[:, type_id], proposal_threshold)
        end_positions = pick_positions(end_probabilities[:, type_id], proposal_threshold)

        if not start_positions or not end_positions:
            continue

        current = []
        for s in start_positions:
            max_e = s + max_span_length - 1
            for e in end_positions:
                if e < s or e > max_e:
                    continue

                start_score = float(start_probabilities[s, type_id])
                end_score = float(end_probabilities[e, type_id])
                span_len = e - s + 1

                boundary_score = (start_score * end_score) ** 0.5
                boundary_score -= SPAN_LENGTH_PENALTY * max(0, span_len - 12) / 50.0

                current.append((s, e + 1, event_type, boundary_score))

        current = sorted(current, key=lambda x: x[3], reverse=True)[:PROPOSAL_PRE_NMS_TOPK]
        candidates.extend(current)

    return candidates


def rerank_candidates_with_span_quality(model, sequence_output_single, candidates, view_type_id: int = 0, section_key_id: int = 0):
    if not candidates:
        return []

    unwrapped = unwrap_model(model)
    span_inputs = [(s, e, EVENT_TYPE_TO_ID[t]) for s, e, t, _ in candidates]

    with torch.no_grad():
        logits = unwrapped.score_spans(sequence_output_single, span_inputs, view_type_id, section_key_id)
        probs = torch.sigmoid(logits).detach().cpu().tolist()

    reranked = []
    section_key = SECTION_KEYS[section_key_id] if 0 <= section_key_id < len(SECTION_KEYS) else "unknown"

    for (s, e, t, boundary_score), quality_prob in zip(candidates, probs):
        prior = max(0.0, section_prior_bonus(t, section_key))
        span_len = e - s
        length_penalty = 0.015 * max(0, span_len - 18) / 18.0
        final_score = 0.62 * quality_prob + 0.28 * max(boundary_score, 0.0) + 0.10 * min(prior * 5.0, 1.0) - length_penalty
        reranked.append((s, e, t, final_score, boundary_score, quality_prob))

    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked


def candidate_effective_score(score, boundary_score, quality_prob, token_start, token_end):
    span_len = token_end - token_start
    return (
        0.72 * score
        + 0.18 * quality_prob
        + 0.10 * boundary_score
        - 0.012 * max(0, span_len - 18) / 18.0
    )


def refine_top_candidates_locally(
    model,
    sequence_output_single,
    start_probabilities,
    end_probabilities,
    candidates,
    valid_positions: Optional[List[int]] = None,
    view_type_id: int = 0,
    section_key_id: int = 0,
    topn: int = 16,
    delta: int = 1,
):
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda x: x[3], reverse=True)
    head = ranked[:topn]
    tail = ranked[topn:]
    refined = []

    unwrapped = unwrap_model(model)
    section_key = SECTION_KEYS[section_key_id] if 0 <= section_key_id < len(SECTION_KEYS) else "unknown"
    seq_len = int(sequence_output_single.size(0))
    valid_position_set = set(int(pos) for pos in valid_positions) if valid_positions is not None else set(range(seq_len))
    if not valid_position_set:
        return candidates
    min_valid_pos = min(valid_position_set)
    max_valid_pos = max(valid_position_set)

    for s, e, t, score, boundary_score, quality_prob in head:
        type_id = EVENT_TYPE_TO_ID[t]
        max_span_length = MAX_SPAN_LENGTH_BY_TYPE[t]

        local_inputs = []
        local_meta = []
        seen = set()

        local_s_min = max(min_valid_pos, s - delta)
        local_s_max = min(max_valid_pos, s + delta)
        local_e_min = max(local_s_min + 1, e - delta)
        local_e_max = min(max_valid_pos + 1, e + delta)

        for ns in range(local_s_min, local_s_max + 1):
            if ns not in valid_position_set:
                continue
            cur_e_min = max(ns + 1, local_e_min)
            cur_e_max = min(local_e_max, ns + max_span_length)
            if cur_e_min > cur_e_max:
                continue
            for ne in range(cur_e_min, cur_e_max + 1):
                if ne <= ns:
                    continue
                if ne > seq_len:
                    continue
                if ne - 1 not in valid_position_set:
                    continue
                key = (ns, ne, t)
                if key in seen:
                    continue
                seen.add(key)
                local_inputs.append((ns, ne, type_id))
                local_meta.append((ns, ne, t))

        if not local_inputs:
            refined.append((s, e, t, score, boundary_score, quality_prob))
            continue

        with torch.no_grad():
            logits = unwrapped.score_spans(sequence_output_single, local_inputs, view_type_id, section_key_id)
            probs = torch.sigmoid(logits).detach().cpu().tolist()

        best_item = (s, e, t, score, boundary_score, quality_prob)
        best_score = score

        original_len = e - s
        for (ns, ne, nt), qp in zip(local_meta, probs):
            start_score = float(start_probabilities[ns, type_id])
            end_score = float(end_probabilities[ne - 1, type_id])
            new_boundary = (start_score * end_score) ** 0.5
            prior = max(0.0, section_prior_bonus(nt, section_key))
            new_len = ne - ns
            change_penalty = 0.020 * abs(new_len - original_len) / max(1, original_len)
            long_penalty = 0.012 * max(0, new_len - 18) / 18.0
            new_score = 0.62 * qp + 0.28 * new_boundary + 0.10 * min(prior * 5.0, 1.0) - change_penalty - long_penalty

            if new_score > best_score + 0.01:
                best_score = new_score
                best_item = (ns, ne, nt, new_score, new_boundary, qp)

        refined.append(best_item)

    refined.extend(tail)
    refined.sort(key=lambda x: x[3], reverse=True)
    return refined


def filter_span_candidates(candidates, text: str, offset_mapping):
    dedup = {}

    for token_start, token_end, event_type, score, boundary_score, quality_prob in candidates:
        info = convert_token_span_to_char_span(text, offset_mapping, (token_start, token_end, event_type))
        span_text = normalize_span_text(info["text"])
        effective_score = candidate_effective_score(score, boundary_score, quality_prob, token_start, token_end)

        if not span_text:
            continue
        if is_heading_term_only(span_text):
            continue
        if quality_prob < 0.20:
            continue
        if effective_score < FINAL_SCORE_THRESHOLD_BY_TYPE[event_type]:
            continue

        key = (token_start, token_end, event_type)
        value = (token_start, token_end, event_type, effective_score, boundary_score, quality_prob, score)

        if key not in dedup or effective_score > dedup[key][3]:
            dedup[key] = value

    selected = []
    for event_type in EVENT_TYPES:
        current = [item for item in dedup.values() if item[2] == event_type]
        current.sort(key=lambda x: x[3], reverse=True)

        kept = []
        for cand in current:
            if len(kept) >= CANDIDATE_LIMIT_CONFIG[event_type]:
                break

            conflict = False
            for old in kept:
                if overlap_ratio_on_shorter_span(cand, old) >= FINAL_OVERLAP_THRESHOLD:
                    conflict = True
                    break

            if not conflict:
                kept.append(cand)

        selected.extend(kept)

    selected.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(s, e, t) for s, e, t, _, _, _, _ in selected]


def cluster_candidates_by_overlap(candidates, overlap_threshold=0.7):
    clusters = []
    candidates = sorted(candidates, key=lambda x: (x["char_start"], x["char_end"], x["type"], -x["score"]))
    for candidate in candidates:
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            if candidate["type"] != anchor["type"]:
                continue
            overlap = max(0, min(candidate["char_end"], anchor["char_end"]) - max(candidate["char_start"], anchor["char_start"]))
            if overlap <= 0:
                continue
            shorter = min(max(1, candidate["char_end"] - candidate["char_start"]), max(1, anchor["char_end"] - anchor["char_start"]))
            if overlap / shorter >= overlap_threshold:
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])
    return clusters


def fuse_cluster_to_node(cluster):
    boundary_score_map = defaultdict(float)
    boundary_text_map = {}
    boundary_view_map = defaultdict(set)
    section_counter = Counter()

    for item in cluster:
        key = (item["char_start"], item["char_end"])
        boundary_score_map[key] += item["score"]
        boundary_text_map[key] = item["text"]
        boundary_view_map[key].add(item["view_id"])
        section_counter[item["section_key"]] += 1

    def boundary_rank(boundary):
        start, end = boundary
        span_len = end - start
        support = len(boundary_view_map[boundary])
        score = boundary_score_map[boundary]
        adjusted = score + 0.03 * support - 0.010 * max(0, span_len - 24) / 24.0
        return (adjusted, support, -span_len)

    best_boundary = max(boundary_score_map.keys(), key=boundary_rank)

    best_start, best_end = best_boundary
    best_text = boundary_text_map[best_boundary]
    support_views = sorted(set(item["view_id"] for item in cluster))
    avg_score = sum(item["score"] for item in cluster) / len(cluster)

    return {
        "start": best_start,
        "end": best_end,
        "text": best_text,
        "type": cluster[0]["type"],
        "score": avg_score,
        "support_views": support_views,
        "support_count": len(support_views),
        "view_consistency_score": len(support_views) / max(len(cluster), 1),
        "section_distribution": {key: value / len(cluster) for key, value in section_counter.items()},
    }


def global_decode_fused_nodes(fused_candidates, final_overlap_threshold=0.80):
    by_type = defaultdict(list)
    for item in fused_candidates:
        by_type[item["type"]].append(item)

    final_nodes = []
    for event_type, items in by_type.items():
        def node_keep_score(x):
            span_len = x["end"] - x["start"]
            support_bonus = 0.03 * min(x.get("support_count", 1), 3)
            consistency_bonus = 0.04 * x.get("view_consistency_score", 0.0)
            long_penalty = 0.012 * max(0, span_len - 24) / 24.0
            return x["score"] + support_bonus + consistency_bonus - long_penalty

        items = sorted(items, key=node_keep_score, reverse=True)
        kept = []

        for item in items:
            keep_score = node_keep_score(item)
            dynamic_threshold = FINAL_SCORE_THRESHOLD_BY_TYPE[event_type]
            if item.get("support_count", 1) <= 1:
                dynamic_threshold += 0.03
            if keep_score < dynamic_threshold:
                continue

            conflict = False
            for old in kept:
                overlap = max(0, min(item["end"], old["end"]) - max(item["start"], old["start"]))
                if overlap <= 0:
                    continue

                shorter = min(
                    max(1, item["end"] - item["start"]),
                    max(1, old["end"] - old["start"]),
                )
                if overlap / shorter >= final_overlap_threshold:
                    conflict = True
                    break

            if conflict:
                continue

            kept.append(item)
            if len(kept) >= CANDIDATE_LIMIT_CONFIG[event_type]:
                break

        final_nodes.extend(kept)

    final_nodes = sorted(final_nodes, key=lambda x: (x["start"], x["end"], x["type"]))
    reindexed = []
    for idx, node in enumerate(final_nodes):
        reindexed.append({
            "node_id": idx,
            "pred_event_id": f"P{idx + 1}",
            "start": node["start"],
            "end": node["end"],
            "text": node["text"],
            "type": node["type"],
            "score": node["score"],
            "support_views": node["support_views"],
            "support_count": node["support_count"],
            "view_consistency_score": node["view_consistency_score"],
            "section_distribution": node["section_distribution"],
        })
    return reindexed


def merge_document_predictions(predictions_by_doc):
    merged = {}
    for doc_id, items in predictions_by_doc.items():
        text = items[0]["text"]
        all_candidates = []
        for item in items:
            all_candidates.extend(item["candidate_nodes"])
        clusters = cluster_candidates_by_overlap(all_candidates, overlap_threshold=0.7)
        fused_candidates = [fuse_cluster_to_node(cluster) for cluster in clusters]
        final_nodes = global_decode_fused_nodes(fused_candidates)
        predicted_char_spans = sorted({(node["start"], node["end"], node["type"]) for node in final_nodes}, key=lambda x: (x[0], x[1], x[2]))
        merged[doc_id] = {
            "text": text,
            "predicted_char_spans": predicted_char_spans,
            "predicted_nodes": final_nodes,
            "candidate_count_before_fusion": len(all_candidates),
            "candidate_count_after_fusion": len(fused_candidates),
        }
    return merged


def compute_span_metrics(predicted_spans_batch, gold_spans_batch):
    pred_total = gold_total = correct_total = 0
    for pred, gold in zip(predicted_spans_batch, gold_spans_batch):
        ps = set(pred)
        gs = set(gold)
        pred_total += len(ps)
        gold_total += len(gs)
        correct_total += len(ps & gs)
    precision = correct_total / pred_total if pred_total else 0.0
    recall = correct_total / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def analyze_recall_errors(predicted_spans_batch, gold_spans_batch):
    stats = {
        "gold_total_spans": 0,
        "gold_hit_exact": 0,
        "gold_miss_no_overlap": 0,
        "gold_miss_type_error": 0,
        "gold_miss_boundary_error": 0,
    }
    for pred, gold in zip(predicted_spans_batch, gold_spans_batch):
        pred_set = set(pred)
        for g in gold:
            stats["gold_total_spans"] += 1
            if g in pred_set:
                stats["gold_hit_exact"] += 1
                continue
            overlapping = [p for p in pred if max(p[0], g[0]) < min(p[1], g[1])]
            if not overlapping:
                stats["gold_miss_no_overlap"] += 1
                continue
            same_type = [p for p in overlapping if p[2] == g[2]]
            if same_type:
                stats["gold_miss_boundary_error"] += 1
            else:
                stats["gold_miss_type_error"] += 1
    total_miss = stats["gold_miss_no_overlap"] + stats["gold_miss_type_error"] + stats["gold_miss_boundary_error"]
    stats["gold_total_miss"] = total_miss
    stats["exact_hit_rate"] = stats["gold_hit_exact"] / stats["gold_total_spans"] if stats["gold_total_spans"] else 0.0
    if total_miss > 0:
        stats["miss_no_overlap_ratio"] = stats["gold_miss_no_overlap"] / total_miss
        stats["miss_type_error_ratio"] = stats["gold_miss_type_error"] / total_miss
        stats["miss_boundary_error_ratio"] = stats["gold_miss_boundary_error"] / total_miss
    else:
        stats["miss_no_overlap_ratio"] = 0.0
        stats["miss_type_error_ratio"] = 0.0
        stats["miss_boundary_error_ratio"] = 0.0
    return stats


def compute_per_type_metrics(predicted_spans_batch, gold_spans_batch):
    results = {}
    for event_type in EVENT_TYPES:
        pred_list = []
        gold_list = []
        for pred, gold in zip(predicted_spans_batch, gold_spans_batch):
            pred_list.append([s for s in pred if s[2] == event_type])
            gold_list.append([s for s in gold if s[2] == event_type])
        results[event_type] = compute_span_metrics(pred_list, gold_list)
    return results


def save_prediction_dump(path: Path, merged_predictions, gold_by_doc, split_name: str, fold_index: int):
    rows = []
    ordered_doc_ids = sorted(merged_predictions.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for doc_id in ordered_doc_ids:
        pred_info = merged_predictions[doc_id]
        text = pred_info["text"]
        predicted_spans = []
        for s, e, t in pred_info["predicted_char_spans"]:
            predicted_spans.append({
                "type": t,
                "text": text[s:e],
                "char_start": s,
                "char_end": e,
            })
        gold_spans = []
        for s, e, t in gold_by_doc.get(doc_id, []):
            gold_spans.append({
                "type": t,
                "text": text[s:e],
                "char_start": s,
                "char_end": e,
            })
        rows.append({
            "doc_id": str(doc_id),
            "text": text,
            "fold_index": int(fold_index),
            "split": split_name,
            "task": PREDICTION_TASK_NAME,
            "prediction_schema": PREDICTION_SCHEMA_NAME,
            "preferred_graph_input_field": PREFERRED_GRAPH_INPUT_FIELD,
            "event_types": list(EVENT_TYPES),
            "predicted_spans": predicted_spans,
            "predicted_nodes": pred_info.get("predicted_nodes", []),
            "candidate_count_before_fusion": pred_info.get("candidate_count_before_fusion", 0),
            "candidate_count_after_fusion": pred_info.get("candidate_count_after_fusion", 0),
            "gold_spans": gold_spans,
        })
    save_jsonl(rows, str(path))


def collect_model_outputs(model, dataloader, positive_weight, span_quality_pos_weight):
    model.eval()
    total_loss = 0.0
    by_doc_predictions = defaultdict(list)
    gold_by_doc = {}
    raw_candidate_hit = 0
    raw_candidate_total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            valid_mask = batch["valid_mask"].to(DEVICE)
            start_labels = batch["start_labels"].to(DEVICE)
            end_labels = batch["end_labels"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                boundary_loss = compute_boundary_loss(
                    outputs["start_logits"],
                    outputs["end_logits"],
                    start_labels,
                    end_labels,
                    valid_mask,
                    positive_weight,
                )
                quality_loss = compute_span_quality_loss(
                    unwrap_model(model),
                    outputs["sequence_output"],
                    batch["span_training_examples"],
                    batch["view_type_ids"],
                    batch["section_key_ids"],
                    span_quality_pos_weight,
                )
                loss = boundary_loss + SPAN_QUALITY_LOSS_WEIGHT * quality_loss
            total_loss += loss.item()

            start_probs = torch.sigmoid(outputs["start_logits"]).detach().cpu()
            end_probs = torch.sigmoid(outputs["end_logits"]).detach().cpu()
            valid_mask_cpu = valid_mask.detach().cpu()
            sequence_output_cpu = outputs["sequence_output"].detach()

            for i in range(start_probs.size(0)):
                valid_positions = torch.nonzero(valid_mask_cpu[i] > 0, as_tuple=False).flatten().tolist()
                if not valid_positions:
                    continue
                seq_end = max(valid_positions) + 1
                text = batch["texts"][i]
                doc_id = batch["doc_ids"][i]
                offset_mapping = batch["offset_mappings"][i]
                raw_candidates = build_span_candidates(
                    start_probs[i, :seq_end, :],
                    end_probs[i, :seq_end, :],
                    valid_positions,
                )
                reranked = rerank_candidates_with_span_quality(
                    model,
                    sequence_output_cpu[i, :seq_end, :],
                    raw_candidates,
                    int(batch["view_type_ids"][i].item()),
                    int(batch["section_key_ids"][i].item()),
                )
                reranked = refine_top_candidates_locally(
                    model,
                    sequence_output_cpu[i, :seq_end, :],
                    start_probs[i, :seq_end, :],
                    end_probs[i, :seq_end, :],
                    reranked,
                    valid_positions,
                    int(batch["view_type_ids"][i].item()),
                    int(batch["section_key_ids"][i].item()),
                )
                token_spans = filter_span_candidates(reranked, text, offset_mapping)
                candidate_nodes = []
                for span in token_spans:
                    info = convert_token_span_to_char_span(text, offset_mapping, span)
                    candidate_nodes.append({
                        "char_start": info["char_start"],
                        "char_end": info["char_end"],
                        "text": info["text"],
                        "type": info["type"],
                        "score": next((cand[3] for cand in reranked if cand[0] == span[0] and cand[1] == span[1] and cand[2] == span[2]), 0.0),
                        "view_id": batch["view_ids"][i],
                        "view_type": batch["view_types"][i],
                        "section_key": batch["primary_section_keys"][i],
                    })
                by_doc_predictions[doc_id].append({
                    "text": text,
                    "candidate_nodes": candidate_nodes,
                })
                gold_by_doc[doc_id] = batch["document_gold_spans"][i]

                raw_candidate_char_spans = {
                    convert_token_span_tuple_to_char_span(text, offset_mapping, (s, e, t))
                    for s, e, t, _ in raw_candidates
                }
                gold_set = set(batch["document_gold_spans"][i])
                raw_candidate_hit += len(raw_candidate_char_spans & gold_set)
                raw_candidate_total += len(gold_set)

    merged = merge_document_predictions(by_doc_predictions)
    ordered_doc_ids = sorted(gold_by_doc.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    predicted_batch = [merged.get(doc_id, {"predicted_char_spans": []})["predicted_char_spans"] for doc_id in ordered_doc_ids]
    gold_batch = [gold_by_doc[doc_id] for doc_id in ordered_doc_ids]

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "predicted_spans_batch": predicted_batch,
        "gold_spans_batch": gold_batch,
        "gold_by_doc": gold_by_doc,
        "merged_predictions": merged,
        "ordered_doc_ids": ordered_doc_ids,
        "raw_candidate_gold_recall": raw_candidate_hit / raw_candidate_total if raw_candidate_total else 0.0,
    }


def evaluate_outputs(outputs, split_name: str, dump_path: Path = None, verbose: bool = True, fold_index: int = -1):
    metrics = compute_span_metrics(outputs["predicted_spans_batch"], outputs["gold_spans_batch"])
    metrics["loss"] = outputs["loss"]
    metrics["error_breakdown"] = analyze_recall_errors(outputs["predicted_spans_batch"], outputs["gold_spans_batch"])
    metrics["per_type_metrics"] = compute_per_type_metrics(outputs["predicted_spans_batch"], outputs["gold_spans_batch"])
    metrics["gold_total"] = sum(len(x) for x in outputs["gold_spans_batch"])
    metrics["raw_candidate_gold_recall"] = outputs.get("raw_candidate_gold_recall", 0.0)
    metrics["fold_index"] = fold_index
    metrics["split"] = split_name
    metrics["prediction_schema"] = PREDICTION_SCHEMA_NAME
    metrics["preferred_graph_input_field"] = PREFERRED_GRAPH_INPUT_FIELD

    if dump_path is not None:
        save_prediction_dump(dump_path, outputs["merged_predictions"], outputs["gold_by_doc"], split_name=split_name, fold_index=fold_index)

    if verbose:
        print("\n================ MULTIVIEW DOCUMENT SPAN EVAL ================")
        print(f"split={split_name} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        print(f"gold_total={metrics['gold_total']} raw_candidate_gold_recall={metrics['raw_candidate_gold_recall']:.4f}")
        print(json.dumps(metrics["error_breakdown"], ensure_ascii=False, indent=2))
        print(json.dumps(metrics["per_type_metrics"], ensure_ascii=False, indent=2))
        print("==================================================================\n")

    return metrics


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def train_one_fold(
    fold_index: int,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    test_samples: List[Dict],
    tokenizer,
    pretrained_model_path,
    run_label: Optional[str] = None,
    record_fold_index: Optional[int] = None,
    export_train_predictions: bool = True,
):
    record_fold_index = fold_index if record_fold_index is None else int(record_fold_index)
    fold_output_dir = OUTPUT_DIR / (run_label or f"fold_{fold_index}")
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = fold_output_dir / "best_model.pt"
    train_predictions_path = fold_output_dir / f"fold_{record_fold_index}_train_event_predictions.jsonl"
    dev_predictions_path = fold_output_dir / f"fold_{record_fold_index}_dev_event_predictions.jsonl"
    test_predictions_path = fold_output_dir / f"fold_{record_fold_index}_test_event_predictions.jsonl"
    train_metrics_path = fold_output_dir / "train_metrics.json"
    test_metrics_path = fold_output_dir / "test_metrics.json"

    train_dataset = DocumentWindowDataset(train_samples, tokenizer)
    dev_dataset = DocumentWindowDataset(dev_samples, tokenizer)
    test_dataset = DocumentWindowDataset(test_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0,
    )
    # Use a non-shuffled loader for reproducible train-split prediction export.
    train_eval_loader = DataLoader(
        train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0,
    )

    positive_weight = build_positive_class_weights(train_dataset)
    span_quality_pos_weight = build_span_quality_pos_weight(train_dataset)

    model = MiddleStableSpanModel(pretrained_model_path, len(EVENT_TYPES), DROPOUT).to(DEVICE)
    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    best_dev_f1 = -1.0
    patience_counter = 0

    display_name = run_label or f"fold_{fold_index}"
    print(f"\n================ MULTIVIEW {display_name} ================")
    print(f"train documents: {len(train_samples)} | train windows: {len(train_dataset)}")
    print(f"dev documents:   {len(dev_samples)} | dev windows:   {len(dev_dataset)}")
    print(f"test documents:  {len(test_samples)} | test windows:  {len(test_dataset)}")
    print("================================================================\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Multiview {display_name} Epoch {epoch + 1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            valid_mask = batch["valid_mask"].to(DEVICE, non_blocking=True)
            start_labels = batch["start_labels"].to(DEVICE, non_blocking=True)
            end_labels = batch["end_labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                boundary_loss = compute_boundary_loss(
                    outputs["start_logits"], outputs["end_logits"],
                    start_labels, end_labels, valid_mask, positive_weight,
                )
                quality_loss = compute_span_quality_loss(
                    unwrap_model(model),
                    outputs["sequence_output"],
                    batch["span_training_examples"],
                    batch["view_type_ids"],
                    batch["section_key_ids"],
                    span_quality_pos_weight,
                )
                loss = boundary_loss + SPAN_QUALITY_LOSS_WEIGHT * quality_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_train_loss / max(1, len(train_loader))
        dev_outputs = collect_model_outputs(model, dev_loader, positive_weight, span_quality_pos_weight)
        dev_metrics = evaluate_outputs(dev_outputs, "dev", None, True, fold_index=record_fold_index)

        print(
            f"[Multiview {display_name}] Epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | dev_loss={dev_metrics['loss']:.4f} | "
            f"dev_f1={dev_metrics['f1']:.4f} | raw_candidate_gold_recall={dev_metrics['raw_candidate_gold_recall']:.4f}"
        )

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            patience_counter = 0
            torch.save({"model_state": unwrap_model(model).state_dict()}, str(best_model_path))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Multiview {display_name}] early stopping triggered")
                break

    checkpoint = torch.load(str(best_model_path), map_location=DEVICE)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])

    # Export train predictions only when they are intended as graph observations.
    # The manuscript protocol uses inner OOF observations for outer train docs.
    if export_train_predictions:
        train_outputs = collect_model_outputs(model, train_eval_loader, positive_weight, span_quality_pos_weight)
        train_metrics = evaluate_outputs(train_outputs, "train", train_predictions_path, True, fold_index=record_fold_index)
        save_json(train_metrics, str(train_metrics_path))
        train_prediction_path_value = str(train_predictions_path)
    else:
        train_metrics = {
            "fold_index": record_fold_index,
            "split": "train",
            "prediction_schema": PREDICTION_SCHEMA_NAME,
            "preferred_graph_input_field": PREFERRED_GRAPH_INPUT_FIELD,
            "exported": False,
            "reason": "outer_train_predictions_are_generated_by_inner_oof",
        }
        train_prediction_path_value = None

    dev_outputs = collect_model_outputs(model, dev_loader, positive_weight, span_quality_pos_weight)
    dev_best_metrics = evaluate_outputs(dev_outputs, "dev", dev_predictions_path, True, fold_index=record_fold_index)
    test_outputs = collect_model_outputs(model, test_loader, positive_weight, span_quality_pos_weight)
    test_metrics = evaluate_outputs(test_outputs, "test", test_predictions_path, True, fold_index=record_fold_index)
    save_json(test_metrics, str(test_metrics_path))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold_index": fold_index,
        "record_fold_index": record_fold_index,
        "run_label": display_name,
        "train_document_count": len(train_samples),
        "dev_document_count": len(dev_samples),
        "test_document_count": len(test_samples),
        "train_window_count": len(train_dataset),
        "dev_window_count": len(dev_dataset),
        "test_window_count": len(test_dataset),
        "dev_best_f1": dev_best_metrics["f1"],
        "train_metrics": train_metrics,
        "dev_best_metrics": dev_best_metrics,
        "train_prediction_path": train_prediction_path_value,
        "dev_prediction_path": str(dev_predictions_path),
        "test_prediction_path": str(test_predictions_path),
        "test_metrics": test_metrics,
    }


def _span_tuple_from_dump(item: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(item["char_start"]), int(item["char_end"]), str(item["type"]))


def evaluate_prediction_records(
    records: List[Dict[str, Any]],
    split_name: str,
    fold_index: int,
    protocol: str,
) -> Dict[str, Any]:
    ordered = sorted(records, key=lambda x: int(x["doc_id"]) if str(x["doc_id"]).isdigit() else str(x["doc_id"]))
    predicted_batch = [
        [_span_tuple_from_dump(item) for item in record.get("predicted_spans", [])]
        for record in ordered
    ]
    gold_batch = [
        [_span_tuple_from_dump(item) for item in record.get("gold_spans", [])]
        for record in ordered
    ]
    metrics = compute_span_metrics(predicted_batch, gold_batch)
    metrics["loss"] = None
    metrics["error_breakdown"] = analyze_recall_errors(predicted_batch, gold_batch)
    metrics["per_type_metrics"] = compute_per_type_metrics(predicted_batch, gold_batch)
    metrics["gold_total"] = sum(len(x) for x in gold_batch)
    metrics["fold_index"] = int(fold_index)
    metrics["split"] = split_name
    metrics["prediction_schema"] = PREDICTION_SCHEMA_NAME
    metrics["preferred_graph_input_field"] = PREFERRED_GRAPH_INPUT_FIELD
    metrics["event_observation_protocol"] = protocol
    return metrics


def build_inner_oof_train_predictions(
    outer_fold_index: int,
    train_samples: List[Dict],
    fold_assignment: Dict[str, int],
    tokenizer,
    pretrained_model_path,
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    protocol = "inner_out_of_fold_within_outer_train"
    outer_fold_dir = OUTPUT_DIR / f"fold_{outer_fold_index}"
    train_predictions_path = outer_fold_dir / f"fold_{outer_fold_index}_train_event_predictions.jsonl"
    train_metrics_path = outer_fold_dir / "train_metrics.json"

    inner_fold_indices = sorted({fold_assignment[str(sample["doc_id"])] for sample in train_samples})
    if len(inner_fold_indices) < 2:
        raise ValueError(
            f"outer fold {outer_fold_index} needs at least two inner train folds for OOF event observations; "
            f"got {inner_fold_indices}"
        )

    combined_records: List[Dict[str, Any]] = []
    inner_summaries: List[Dict[str, Any]] = []
    for heldout_fold in inner_fold_indices:
        heldout_samples = [
            sample for sample in train_samples
            if fold_assignment[str(sample["doc_id"])] == heldout_fold
        ]
        inner_train_samples = [
            sample for sample in train_samples
            if fold_assignment[str(sample["doc_id"])] != heldout_fold
        ]
        if not heldout_samples or not inner_train_samples:
            raise ValueError(
                f"outer fold {outer_fold_index}, inner heldout fold {heldout_fold} has an empty split"
            )

        inner_train_fold_indices = sorted(set(inner_fold_indices) - {heldout_fold})
        inner_result = train_one_fold(
            fold_index=outer_fold_index,
            train_samples=inner_train_samples,
            dev_samples=heldout_samples,
            test_samples=heldout_samples,
            tokenizer=tokenizer,
            pretrained_model_path=pretrained_model_path,
            run_label=f"fold_{outer_fold_index}/inner_event_oof/heldout_fold_{heldout_fold}",
            record_fold_index=outer_fold_index,
            export_train_predictions=False,
        )
        inner_records = load_jsonl(str(inner_result["test_prediction_path"]))
        for record in inner_records:
            record["fold_index"] = int(outer_fold_index)
            record["split"] = "train"
            record["event_observation_protocol"] = protocol
            record["inner_heldout_fold_index"] = int(heldout_fold)
            record["inner_train_fold_indices"] = inner_train_fold_indices
            record["source_prediction_path"] = inner_result["test_prediction_path"]
        combined_records.extend(inner_records)
        inner_summaries.append(
            {
                "heldout_fold_index": int(heldout_fold),
                "inner_train_fold_indices": inner_train_fold_indices,
                "heldout_document_count": len(heldout_samples),
                "inner_train_document_count": len(inner_train_samples),
                "prediction_path": inner_result["test_prediction_path"],
                "heldout_metrics": inner_result["test_metrics"],
            }
        )

    expected_doc_ids = {str(sample["doc_id"]) for sample in train_samples}
    observed_doc_ids = [str(record["doc_id"]) for record in combined_records]
    duplicate_doc_ids = sorted({doc_id for doc_id in observed_doc_ids if observed_doc_ids.count(doc_id) > 1})
    missing_doc_ids = sorted(expected_doc_ids - set(observed_doc_ids))
    extra_doc_ids = sorted(set(observed_doc_ids) - expected_doc_ids)
    if duplicate_doc_ids or missing_doc_ids or extra_doc_ids:
        raise RuntimeError(
            "inner OOF train prediction coverage mismatch: "
            + json.dumps(
                {
                    "outer_fold_index": outer_fold_index,
                    "duplicate_doc_ids": duplicate_doc_ids[:20],
                    "missing_doc_ids": missing_doc_ids[:20],
                    "extra_doc_ids": extra_doc_ids[:20],
                },
                ensure_ascii=False,
            )
        )

    combined_records.sort(key=lambda x: int(x["doc_id"]) if str(x["doc_id"]).isdigit() else str(x["doc_id"]))
    save_jsonl(combined_records, str(train_predictions_path))
    train_metrics = evaluate_prediction_records(
        combined_records,
        split_name="train",
        fold_index=outer_fold_index,
        protocol=protocol,
    )
    train_metrics["inner_oof_fold_count"] = len(inner_summaries)
    train_metrics["inner_oof_summaries"] = inner_summaries
    save_json(train_metrics, str(train_metrics_path))
    return train_metrics, str(train_predictions_path), inner_summaries


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    resolved_pretrained_path = resolve_local_pretrained_path(PRETRAINED_MODEL_NAME)
    print(f"Using local pretrained path: {resolved_pretrained_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_pretrained_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    samples = load_jsonl(str(INPUT_SAMPLE_PATH))
    assert_fast_tokenizer(tokenizer)
    validate_tokenizer_offset_mapping(tokenizer, samples)
    fold_payload = load_fold_assignment(INPUT_FOLD_PATH)
    num_folds = int(fold_payload["num_folds"])
    fold_assignment = {str(k): int(v) for k, v in fold_payload["fold_assignment"].items()}

    if num_folds != 5:
        raise ValueError(f"当前最终流程要求 5 折，实际 fold 数为 {num_folds}")

    all_fold_results = []
    fold_indices = list(range(num_folds))

    for test_fold in fold_indices:
        dev_fold = (test_fold + 1) % num_folds
        train_samples, dev_samples, test_samples = split_samples_by_fold(
            samples,
            fold_assignment,
            test_fold,
            dev_fold,
        )
        fold_result = train_one_fold(
            test_fold,
            train_samples,
            dev_samples,
            test_samples,
            tokenizer,
            resolved_pretrained_path,
            export_train_predictions=False,
        )
        train_metrics, train_prediction_path, inner_summaries = build_inner_oof_train_predictions(
            outer_fold_index=test_fold,
            train_samples=train_samples,
            fold_assignment=fold_assignment,
            tokenizer=tokenizer,
            pretrained_model_path=resolved_pretrained_path,
        )
        fold_result["train_metrics"] = train_metrics
        fold_result["train_prediction_path"] = train_prediction_path
        fold_result["train_prediction_protocol"] = "inner_out_of_fold_within_outer_train"
        fold_result["train_inner_oof_summaries"] = inner_summaries
        fold_result["dev_fold_index"] = dev_fold
        all_fold_results.append(fold_result)

    precision_values = [r["test_metrics"]["precision"] for r in all_fold_results]
    recall_values = [r["test_metrics"]["recall"] for r in all_fold_results]
    f1_values = [r["test_metrics"]["f1"] for r in all_fold_results]

    summary = {
        "num_folds": num_folds,
        "evaluation_mode": "five_fold_cross_validation",
        "prediction_schema": PREDICTION_SCHEMA_NAME,
        "preferred_graph_input_field": PREFERRED_GRAPH_INPUT_FIELD,
        "event_observation_protocol": {
            "outer_train": "inner_out_of_fold_within_outer_train",
            "outer_dev": "event_model_trained_on_outer_train_folds",
            "outer_test": "event_model_trained_on_outer_train_folds",
        },
        "fold_results": all_fold_results,
        "aggregate_metrics": {
            "precision_mean": mean(precision_values),
            "precision_std": sample_std(precision_values),
            "recall_mean": mean(recall_values),
            "recall_std": sample_std(recall_values),
            "f1_mean": mean(f1_values),
            "f1_std": sample_std(f1_values),
        },
    }

    save_json(summary, str(SUMMARY_PATH))
    print("\n================ MULTIVIEW CROSS VALIDATION SUMMARY ================")
    print(json.dumps(summary["aggregate_metrics"], ensure_ascii=False, indent=2))
    print(f"saved summary to: {SUMMARY_PATH}")
    print("========================================================================\n")


if __name__ == "__main__":
    main()
