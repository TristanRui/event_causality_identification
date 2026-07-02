from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    """Find the project root from the current script path."""
    current = start.resolve()
    candidates = [current.parent] + list(current.parent.parents)
    for p in candidates:
        if (p / "config.py").exists() and (p / "utils").exists() and (p / "graph").exists():
            return p
    return start.resolve().parents[1]


PROJECT_DIR = find_project_root(Path(__file__))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import RANDOM_SEED
from utils.io_utils import load_jsonl, save_json
from utils.seed_utils import set_seed
from graph.graph_schema import adapt_graph_sample, has_target, TARGET_KEYS
from graph.graph_dataset import (
    RelationGraphDataset,
    assert_fast_tokenizer,
    build_category_vocab,
    collate_graph_batch,
    load_fold_assignment,
    split_samples_by_fold,
    count_labeled_features,
)
from graph.graph_core import RelationGraphModel, RELATION_TYPES
from graph.graph_metrics import (
    build_relation_class_weights,
    collect_model_outputs,
    compute_relation_loss,
)


# Keep model loading offline and avoid background Hub calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


INPUT_GRAPH_SAMPLE_ROOT = Path(
    os.environ.get(
        "INPUT_GRAPH_SAMPLE_ROOT",
        str(PROJECT_DIR / "data" / "processed" / "relation_graph_samples"),
    )
)
INPUT_FOLD_PATH = Path(
    os.environ.get(
        "INPUT_FOLD_PATH",
        str(PROJECT_DIR / "data" / "processed" / "event_document_folds.json"),
    )
)

SAMPLE_LOAD_MODE = os.environ.get("SAMPLE_LOAD_MODE", "direct_split").strip()

NODE_SOURCE = os.environ.get("NODE_SOURCE", "predicted_events")
TRAIN_TARGET_KEY = os.environ.get("TRAIN_TARGET_KEY", "train_strict_exact_unmatched_ignore")
EARLY_STOP_TARGET_KEY = os.environ.get("EARLY_STOP_TARGET_KEY", "strict_exact")
TEST_TARGET_KEYS = [
    item.strip()
    for item in os.environ.get("TEST_TARGET_KEYS", "strict_exact").split(",")
    if item.strip()
]

OUTPUT_DIR = Path(
    os.environ.get(
        "CHECKPOINT_ROOT",
        str(PROJECT_DIR / "outputs" / "relation_model"),
    )
)
SUMMARY_PATH = OUTPUT_DIR / "cross_validation_summary.json"


PRETRAINED_MODEL_NAME = os.environ.get("PRETRAINED_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")
WINDOW_TOKEN_LENGTH = int(os.environ.get("WINDOW_TOKEN_LENGTH", "448"))
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "192"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", str(BATCH_SIZE)))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "12"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
CONSTRAINT_LOSS_WEIGHT = float(os.environ.get("CONSTRAINT_LOSS_WEIGHT", "0.1"))
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", "5"))
RUN_FOLDS = [
    int(item.strip())
    for item in os.environ.get("RUN_FOLDS", ",".join(str(i) for i in range(NUM_FOLDS))).split(",")
    if item.strip()
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = os.environ.get("USE_AMP", "1").strip() == "1" and DEVICE.type == "cuda"


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((x - avg) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    """Resolve a Hugging Face repo id to a local snapshot before loading."""
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


def make_grad_scaler():
    if not USE_AMP:
        return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context():
    if not USE_AMP:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def validate_target_key(target_key: str) -> None:
    if target_key not in TARGET_KEYS:
        raise ValueError(
            f"当前 graph_schema.TARGET_KEYS 不支持 target_key={target_key}。"
            f"已支持的 key: {list(TARGET_KEYS)}。"
            "请确认 04 输出的 targets 字段和 graph_schema.py 的 TARGET_KEYS 已同步。"
        )


def load_graph_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到关系图样本文件: {path}")
    raw_samples = load_jsonl(str(path))
    samples = [adapt_graph_sample(item) for item in raw_samples]
    return samples


def resolve_graph_sample_path(root: Path, fold_index: int, split_name: str) -> Optional[Path]:
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


def load_oof_test_pool(root: Path, num_folds: int) -> List[Dict[str, Any]]:
    """Load test graph samples from all folds as a one-record-per-document OOF pool."""
    all_samples: List[Dict[str, Any]] = []
    missing_paths: List[str] = []
    seen_doc_ids: set[str] = set()
    duplicate_doc_ids: List[str] = []

    for fold_index in range(num_folds):
        path = resolve_graph_sample_path(root, fold_index, "test")
        if path is None:
            missing_paths.append(f"fold={fold_index}, split=test")
            continue
        fold_samples = load_graph_samples(path)
        for sample in fold_samples:
            doc_id = str(sample["doc_id"])
            if doc_id in seen_doc_ids:
                duplicate_doc_ids.append(doc_id)
                continue
            seen_doc_ids.add(doc_id)
            all_samples.append(sample)

    if missing_paths:
        raise FileNotFoundError(
            "OOF 模式要求每个 fold 都有 test graph sample 文件，缺失: " + ", ".join(missing_paths)
        )
    if duplicate_doc_ids:
        raise RuntimeError(
            "OOF test 节点池中发现重复 doc_id，说明 04 的输出或 fold 划分存在重复: "
            + json.dumps(sorted(set(duplicate_doc_ids))[:50], ensure_ascii=False)
        )
    if not all_samples:
        raise RuntimeError(f"没有从 {root} 读取到任何 OOF test graph samples")
    return all_samples


def validate_fold_coverage(samples: List[Dict[str, Any]], fold_assignment: Dict[str, int]) -> None:
    sample_doc_ids = {str(item["doc_id"]) for item in samples}
    fold_doc_ids = set(str(k) for k in fold_assignment.keys())
    missing_in_samples = sorted(fold_doc_ids - sample_doc_ids)
    missing_in_fold = sorted(sample_doc_ids - fold_doc_ids)
    if missing_in_samples:
        raise RuntimeError(
            "fold 文件中有文档没有对应的 OOF graph sample，前 20 个: "
            + json.dumps(missing_in_samples[:20], ensure_ascii=False)
        )
    if missing_in_fold:
        raise RuntimeError(
            "graph sample 中有文档不在 fold 文件中，前 20 个: "
            + json.dumps(missing_in_fold[:20], ensure_ascii=False)
        )


def load_direct_split_for_fold(root: Path, fold_index: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load train/dev/test graph samples for one outer fold."""
    split_samples: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ["train", "dev", "test"]:
        path = resolve_graph_sample_path(root, fold_index, split_name)
        if path is None:
            raise FileNotFoundError(f"direct_split 模式缺少 fold={fold_index}, split={split_name} 的样本文件")
        split_samples[split_name] = load_graph_samples(path)
    return split_samples["train"], split_samples["dev"], split_samples["test"]


def target_available_count(samples: Iterable[Dict[str, Any]], target_key: str) -> int:
    count = 0
    for sample in samples:
        try:
            if has_target(sample, target_key=target_key):
                count += 1
        except Exception:
            continue
    return count


def validate_samples_have_target(samples: List[Dict[str, Any]], target_key: str, split_name: str) -> None:
    available = target_available_count(samples, target_key)
    if available <= 0:
        raise RuntimeError(
            f"{split_name} split 中没有任何样本包含 target_key={target_key}。"
            "请检查 04 输出的 targets 字段，或调整 TRAIN_TARGET_KEY/EARLY_STOP_TARGET_KEY/TEST_TARGET_KEYS。"
        )


def build_loader(dataset: RelationGraphDataset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graph_batch,
    )


def compute_regularization_loss(model: RelationGraphModel, pair_logits: List[torch.Tensor], role_masks: List[torch.Tensor]) -> torch.Tensor:
    if CONSTRAINT_LOSS_WEIGHT <= 0:
        return torch.tensor(0.0, device=DEVICE)

    reg_loss = torch.tensor(0.0, device=DEVICE)
    valid_docs = 0
    for logits, role_mask in zip(pair_logits, role_masks):
        if logits.numel() == 0:
            continue
        role_mask = role_mask.to(logits.device)
        probs = torch.softmax(logits.masked_fill(role_mask <= 0, -1e4), dim=-1)
        reg_loss = reg_loss + model.constraint_regularizer(probs)
        valid_docs += 1
    return reg_loss / max(valid_docs, 1)


def build_model(pretrained_model_path: str, category_count: int) -> RelationGraphModel:
    return RelationGraphModel(
        pretrained_model_name=pretrained_model_path,
        category_count=category_count,
        dropout=DROPOUT,
    ).to(DEVICE)


def save_checkpoint(
    path: Path,
    model: RelationGraphModel,
    category_to_id: Dict[str, int],
    fold_index: int,
    dev_fold_index: int,
    epoch: int,
    dev_metrics: Dict[str, Any],
    pretrained_model_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "category_to_id": category_to_id,
        "fold_index": fold_index,
        "dev_fold_index": dev_fold_index,
        "epoch": epoch,
        "dev_metrics": dev_metrics,
        "node_source": NODE_SOURCE,
        "sample_load_mode": SAMPLE_LOAD_MODE,
        "train_target_key": TRAIN_TARGET_KEY,
        "early_stop_target_key": EARLY_STOP_TARGET_KEY,
        "test_target_keys": TEST_TARGET_KEYS,
        "pretrained_model_name": PRETRAINED_MODEL_NAME,
        "resolved_pretrained_model_path": pretrained_model_path,
        "window_token_length": WINDOW_TOKEN_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "dropout": DROPOUT,
        "relation_none_class_weight_floor": os.environ.get("RELATION_NONE_CLASS_WEIGHT_FLOOR", "0.10"),
        "relation_none_class_weight_cap": os.environ.get("RELATION_NONE_CLASS_WEIGHT_CAP", "0"),
        "relation_class_weight_overrides": os.environ.get("RELATION_CLASS_WEIGHT_OVERRIDES", ""),
        "relation_graph_training": "five_fold_outer_split_predicted_observations",
    }
    torch.save(checkpoint, str(path))


def train_one_fold(
    fold_index: int,
    train_samples: List[Dict[str, Any]],
    dev_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    tokenizer,
    pretrained_model_path: str,
    category_to_id: Dict[str, int],
) -> Dict[str, Any]:
    dev_fold_index = (fold_index + 1) % NUM_FOLDS
    fold_output_dir = OUTPUT_DIR / f"fold_{fold_index}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_output_dir / "best_model.pt"
    test_metrics_path = fold_output_dir / "test_metrics.json"
    fold_summary_path = fold_output_dir / "fold_summary.json"

    validate_samples_have_target(train_samples, TRAIN_TARGET_KEY, "train")
    validate_samples_have_target(dev_samples, EARLY_STOP_TARGET_KEY, "dev")
    for target_key in TEST_TARGET_KEYS:
        validate_samples_have_target(test_samples, target_key, f"test/{target_key}")

    train_dataset = RelationGraphDataset(
        train_samples,
        tokenizer,
        category_to_id,
        target_key=TRAIN_TARGET_KEY,
        window_token_length=WINDOW_TOKEN_LENGTH,
        window_stride=WINDOW_STRIDE,
    )
    dev_dataset = RelationGraphDataset(
        dev_samples,
        tokenizer,
        category_to_id,
        target_key=EARLY_STOP_TARGET_KEY,
        window_token_length=WINDOW_TOKEN_LENGTH,
        window_stride=WINDOW_STRIDE,
    )

    labeled_train_features = count_labeled_features(train_dataset.features)
    if labeled_train_features <= 0:
        raise RuntimeError(
            f"fold {fold_index} 的训练集没有可用 label。请检查 TRAIN_TARGET_KEY={TRAIN_TARGET_KEY}。"
        )

    train_loader = build_loader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader = build_loader(dev_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE)
    class_weights = build_relation_class_weights(train_dataset)

    model = build_model(pretrained_model_path=pretrained_model_path, category_count=len(category_to_id))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader) * NUM_EPOCHS, 1)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = make_grad_scaler()

    best_dev_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    print(f"\n================ RELATION GRAPH FOLD {fold_index} ================")
    print(f"sample_load_mode: {SAMPLE_LOAD_MODE}")
    print(f"train_target_key: {TRAIN_TARGET_KEY}")
    print(f"early_stop_target_key: {EARLY_STOP_TARGET_KEY}")
    print(f"train docs: {len(train_samples)} | labeled features: {labeled_train_features}")
    print(f"dev docs:   {len(dev_samples)}")
    print(f"test docs:  {len(test_samples)}")
    print(
        "class_weights: "
        + ", ".join(f"{name}={float(weight):.4f}" for name, weight in zip(RELATION_TYPES, class_weights))
    )
    print("===============================================================\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_edge_loss = 0.0
        total_reg_loss = 0.0

        progress = tqdm(train_loader, desc=f"Fold {fold_index} Epoch {epoch}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                outputs = model(batch)
                edge_loss = compute_relation_loss(
                    outputs["pair_logits"],
                    batch["target_matrices"],
                    batch["role_constraint_masks"],
                    class_weights,
                    ignore_labels=batch.get("ignore_labels"),
                    device=DEVICE,
                )
                reg_loss = compute_regularization_loss(model, outputs["pair_logits"], batch["role_constraint_masks"])
                loss = edge_loss + CONSTRAINT_LOSS_WEIGHT * reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += float(loss.detach().item())
            total_edge_loss += float(edge_loss.detach().item())
            total_reg_loss += float(reg_loss.detach().item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_train_loss / max(len(train_loader), 1)
        train_edge_loss = total_edge_loss / max(len(train_loader), 1)
        train_reg_loss = total_reg_loss / max(len(train_loader), 1)
        dev_metrics = collect_model_outputs(model, dev_loader, class_weights, device=DEVICE)
        dev_f1 = float(dev_metrics.get("f1", 0.0))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_edge_loss": train_edge_loss,
            "train_reg_loss": train_reg_loss,
            "dev_metrics": dev_metrics,
        }
        history.append(epoch_record)

        print(
            f"[Fold {fold_index}] Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | edge_loss={train_edge_loss:.4f} | reg_loss={train_reg_loss:.4f} | "
            f"dev_loss={dev_metrics.get('loss', 0.0):.4f} | dev_f1={dev_f1:.4f} | "
            f"dev_violation={dev_metrics.get('violation_count', 0)}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                best_model_path,
                model=model,
                category_to_id=category_to_id,
                fold_index=fold_index,
                dev_fold_index=dev_fold_index,
                epoch=epoch,
                dev_metrics=dev_metrics,
                pretrained_model_path=pretrained_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Fold {fold_index}] early stopping triggered at epoch {epoch}")
                break

    if not best_model_path.exists():
        raise RuntimeError(f"fold {fold_index} 没有保存 best_model.pt，训练过程可能未正常产生 dev 指标")

    checkpoint = torch.load(str(best_model_path), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_metrics_by_target: Dict[str, Dict[str, Any]] = {}
    for target_key in TEST_TARGET_KEYS:
        test_dataset = RelationGraphDataset(
            test_samples,
            tokenizer,
            category_to_id,
            target_key=target_key,
            window_token_length=WINDOW_TOKEN_LENGTH,
            window_stride=WINDOW_STRIDE,
        )
        test_loader = build_loader(test_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE)
        test_metrics_by_target[target_key] = collect_model_outputs(
            model,
            test_loader,
            class_weights,
            device=DEVICE,
        )

    save_json(test_metrics_by_target, str(test_metrics_path))

    fold_summary = {
        "fold_index": fold_index,
        "dev_fold_index": dev_fold_index,
        "sample_load_mode": SAMPLE_LOAD_MODE,
        "train_document_count": len(train_samples),
        "dev_document_count": len(dev_samples),
        "test_document_count": len(test_samples),
        "train_target_key": TRAIN_TARGET_KEY,
        "early_stop_target_key": EARLY_STOP_TARGET_KEY,
        "test_target_keys": TEST_TARGET_KEYS,
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "best_model_path": str(best_model_path),
        "test_metrics_by_target": test_metrics_by_target,
        "history": history,
    }
    save_json(fold_summary, str(fold_summary_path))
    return fold_summary


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n================ RELATION GRAPH TRAINING CONFIG ================")
    print(f"input_graph_sample_root: {INPUT_GRAPH_SAMPLE_ROOT}")
    print(f"sample_load_mode: {SAMPLE_LOAD_MODE}")
    print(f"run_folds: {RUN_FOLDS}")
    print(f"checkpoint_root: {OUTPUT_DIR}")
    print(f"train_target_key: {TRAIN_TARGET_KEY}")
    print(f"early_stop_target_key: {EARLY_STOP_TARGET_KEY}")
    print(f"test_target_keys: {TEST_TARGET_KEYS}")
    print("===============================================================\n")

    if NUM_FOLDS != 5 or RUN_FOLDS != [0, 1, 2, 3, 4]:
        raise ValueError(f"当前最终流程要求完整 5 折运行，NUM_FOLDS={NUM_FOLDS}, RUN_FOLDS={RUN_FOLDS}")

    validate_target_key(TRAIN_TARGET_KEY)
    validate_target_key(EARLY_STOP_TARGET_KEY)
    for key in TEST_TARGET_KEYS:
        validate_target_key(key)

    pretrained_model_path = resolve_local_pretrained_path(PRETRAINED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    assert_fast_tokenizer(tokenizer)

    fold_assignment = load_fold_assignment(INPUT_FOLD_PATH)

    all_oof_samples: Optional[List[Dict[str, Any]]] = None
    if SAMPLE_LOAD_MODE == "oof_test_pool":
        all_oof_samples = load_oof_test_pool(INPUT_GRAPH_SAMPLE_ROOT, NUM_FOLDS)
        validate_fold_coverage(all_oof_samples, fold_assignment)
        category_to_id = build_category_vocab(all_oof_samples)
    elif SAMPLE_LOAD_MODE == "direct_split":
        # Build one shared category vocabulary so fold checkpoints share embedding dimensions.
        pooled_samples: List[Dict[str, Any]] = []
        for fold_index in RUN_FOLDS:
            for split_name in ["train", "dev", "test"]:
                path = resolve_graph_sample_path(INPUT_GRAPH_SAMPLE_ROOT, fold_index, split_name)
                if path is not None:
                    pooled_samples.extend(load_graph_samples(path))
        if not pooled_samples:
            raise RuntimeError(f"direct_split 模式没有在 {INPUT_GRAPH_SAMPLE_ROOT} 下读取到任何样本")
        category_to_id = build_category_vocab(pooled_samples)
    else:
        raise ValueError("SAMPLE_LOAD_MODE 只能是 oof_test_pool 或 direct_split")

    all_fold_results: List[Dict[str, Any]] = []
    for test_fold in RUN_FOLDS:
        dev_fold = (test_fold + 1) % NUM_FOLDS
        if SAMPLE_LOAD_MODE == "oof_test_pool":
            assert all_oof_samples is not None
            train_samples, dev_samples, test_samples = split_samples_by_fold(
                all_oof_samples,
                fold_assignment,
                test_fold=test_fold,
                dev_fold=dev_fold,
            )
        else:
            train_samples, dev_samples, test_samples = load_direct_split_for_fold(
                INPUT_GRAPH_SAMPLE_ROOT,
                fold_index=test_fold,
            )

        fold_result = train_one_fold(
            fold_index=test_fold,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
            tokenizer=tokenizer,
            pretrained_model_path=pretrained_model_path,
            category_to_id=category_to_id,
        )
        all_fold_results.append(fold_result)

    aggregate_by_target: Dict[str, Dict[str, float]] = {}
    for target_key in TEST_TARGET_KEYS:
        precision_values = [
            float(result["test_metrics_by_target"].get(target_key, {}).get("precision", 0.0))
            for result in all_fold_results
        ]
        recall_values = [
            float(result["test_metrics_by_target"].get(target_key, {}).get("recall", 0.0))
            for result in all_fold_results
        ]
        f1_values = [
            float(result["test_metrics_by_target"].get(target_key, {}).get("f1", 0.0))
            for result in all_fold_results
        ]
        aggregate_by_target[target_key] = {
            "precision_mean": mean(precision_values),
            "precision_std": sample_std(precision_values),
            "recall_mean": mean(recall_values),
            "recall_std": sample_std(recall_values),
            "f1_mean": mean(f1_values),
            "f1_std": sample_std(f1_values),
        }

    summary = {
        "input_graph_sample_root": str(INPUT_GRAPH_SAMPLE_ROOT),
        "input_fold_path": str(INPUT_FOLD_PATH),
        "output_dir": str(OUTPUT_DIR),
        "sample_load_mode": SAMPLE_LOAD_MODE,
        "node_source": NODE_SOURCE,
        "train_target_key": TRAIN_TARGET_KEY,
        "early_stop_target_key": EARLY_STOP_TARGET_KEY,
        "test_target_keys": TEST_TARGET_KEYS,
        "num_folds": NUM_FOLDS,
        "run_folds": RUN_FOLDS,
        "pretrained_model_name": PRETRAINED_MODEL_NAME,
        "resolved_pretrained_model_path": pretrained_model_path,
        "window_token_length": WINDOW_TOKEN_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "dropout": DROPOUT,
        "patience": PATIENCE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "constraint_loss_weight": CONSTRAINT_LOSS_WEIGHT,
        "relation_none_class_weight_floor": os.environ.get("RELATION_NONE_CLASS_WEIGHT_FLOOR", "0.10"),
        "relation_none_class_weight_cap": os.environ.get("RELATION_NONE_CLASS_WEIGHT_CAP", "0"),
        "relation_class_weight_overrides": os.environ.get("RELATION_CLASS_WEIGHT_OVERRIDES", ""),
        "use_amp": USE_AMP,
        "device": str(DEVICE),
        "category_to_id": category_to_id,
        "fold_results": all_fold_results,
        "aggregate_metrics_by_target": aggregate_by_target,
    }
    save_json(summary, str(SUMMARY_PATH))

    print("\n================ RELATION GRAPH TRAINING SUMMARY ================")
    print(json.dumps(aggregate_by_target, ensure_ascii=False, indent=2))
    print(f"saved summary to: {SUMMARY_PATH}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
