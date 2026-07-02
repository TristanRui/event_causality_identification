from typing import List, Tuple, Set, Dict


def extract_entities_from_bio(label_seq: List[str]) -> Set[Tuple[int, int, str]]:
    """Extract spans from BIO labels with tolerant handling of malformed I-tags."""
    entities = set()
    start = None
    ent_type = None

    for idx, label in enumerate(label_seq + ["O"]):
        if label == "O":
            if start is not None:
                entities.add((start, idx, ent_type))
                start = None
                ent_type = None
            continue

        if "-" not in label:
            if start is not None:
                entities.add((start, idx, ent_type))
                start = None
                ent_type = None
            continue

        prefix, cur_type = label.split("-", 1)

        if prefix == "B":
            if start is not None:
                entities.add((start, idx, ent_type))
            start = idx
            ent_type = cur_type

        elif prefix == "I":
            if start is None:
                start = idx
                ent_type = cur_type
            elif cur_type != ent_type:
                entities.add((start, idx, ent_type))
                start = idx
                ent_type = cur_type

        else:
            if start is not None:
                entities.add((start, idx, ent_type))
                start = None
                ent_type = None

    return entities


def _span_overlap(span_a: Tuple[int, int, str], span_b: Tuple[int, int, str]) -> bool:
    a_start, a_end, _ = span_a
    b_start, b_end, _ = span_b
    return max(a_start, b_start) < min(a_end, b_end)


def analyze_recall_errors(
    pred_label_seqs: List[List[str]],
    gold_label_seqs: List[List[str]]
) -> Dict[str, float]:
    """Analyze recall misses using the same span extraction policy as event F1."""
    stats = {
        "gold_total_spans": 0,
        "gold_hit_exact": 0,
        "gold_miss_no_overlap": 0,
        "gold_miss_type_error": 0,
        "gold_miss_boundary_error": 0
    }

    for pred_seq, gold_seq in zip(pred_label_seqs, gold_label_seqs):
        pred_entities = extract_entities_from_bio(pred_seq)
        gold_entities = extract_entities_from_bio(gold_seq)

        for gold_ent in gold_entities:
            stats["gold_total_spans"] += 1

            if gold_ent in pred_entities:
                stats["gold_hit_exact"] += 1
                continue

            overlap_preds = [p for p in pred_entities if _span_overlap(p, gold_ent)]

            if not overlap_preds:
                stats["gold_miss_no_overlap"] += 1
                continue

            same_type_overlap = [p for p in overlap_preds if p[2] == gold_ent[2]]

            if same_type_overlap:
                stats["gold_miss_boundary_error"] += 1
            else:
                stats["gold_miss_type_error"] += 1

    total_miss = (
        stats["gold_miss_no_overlap"] +
        stats["gold_miss_type_error"] +
        stats["gold_miss_boundary_error"]
    )
    stats["gold_total_miss"] = total_miss

    if stats["gold_total_spans"] > 0:
        stats["exact_hit_rate"] = stats["gold_hit_exact"] / stats["gold_total_spans"]
    else:
        stats["exact_hit_rate"] = 0.0

    if total_miss > 0:
        stats["miss_no_overlap_ratio"] = stats["gold_miss_no_overlap"] / total_miss
        stats["miss_type_error_ratio"] = stats["gold_miss_type_error"] / total_miss
        stats["miss_boundary_error_ratio"] = stats["gold_miss_boundary_error"] / total_miss
    else:
        stats["miss_no_overlap_ratio"] = 0.0
        stats["miss_type_error_ratio"] = 0.0
        stats["miss_boundary_error_ratio"] = 0.0

    return stats


def compute_event_f1(pred_seqs: List[List[str]], gold_seqs: List[List[str]]) -> Dict[str, float]:
    """Compute exact span-level event F1."""
    pred_total = 0
    gold_total = 0
    correct_total = 0

    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs):
        pred_entities = extract_entities_from_bio(pred_seq)
        gold_entities = extract_entities_from_bio(gold_seq)

        pred_total += len(pred_entities)
        gold_total += len(gold_entities)
        correct_total += len(pred_entities & gold_entities)

    precision = correct_total / pred_total if pred_total > 0 else 0.0
    recall = correct_total / gold_total if gold_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_relation_f1(
    pred_labels: List[int],
    gold_labels: List[int],
    none_label_id: int = 0
) -> Dict[str, float]:
    """Compute positive-label relation F1."""
    tp = 0
    fp = 0
    fn = 0

    for pred, gold in zip(pred_labels, gold_labels):
        if pred != none_label_id and gold != none_label_id:
            if pred == gold:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif pred != none_label_id and gold == none_label_id:
            fp += 1
        elif pred == none_label_id and gold != none_label_id:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
