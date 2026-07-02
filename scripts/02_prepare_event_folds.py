import sys
import os
import json
import random
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import RANDOM_SEED
from utils.io_utils import load_jsonl, save_jsonl, save_json
from utils.text_utils import ensure_sentences

INPUT_ANNOTATION_PATH = PROJECT_DIR / 'data' / 'annotations' / 'gold_sample.jsonl'
OUTPUT_SAMPLE_PATH = PROJECT_DIR / 'data' / 'processed' / 'event_samples_all.jsonl'
OUTPUT_FOLD_PATH = PROJECT_DIR / 'data' / 'processed' / 'event_document_folds.json'
OUTPUT_SUMMARY_PATH = PROJECT_DIR / 'data' / 'processed' / 'event_fold_summary.json'

NUM_FOLDS = 5
NEGATIVE_SAMPLE_KEEP_RATIO = 1.0

# Reuse an existing fold assignment to keep train/dev/test document ids fixed.
FIX_FOLD_ASSIGNMENT = True

# Prefer a frozen fold assignment; fall back to OUTPUT_FOLD_PATH if absent.
FROZEN_FOLD_PATH = PROJECT_DIR / 'data' / 'processed' / 'event_document_folds_frozen.json'

# Fail fast if the current gold document ids differ from the frozen assignment.
STRICT_DOC_ID_CHECK = True

VALIDATE_MANUSCRIPT_DATASET = os.environ.get('VALIDATE_MANUSCRIPT_DATASET', '1').strip() != '0'
EXPECTED_DOCUMENT_COUNT = 286
EXPECTED_EVENT_TYPE_COUNTS = {
    'PHENOMENON': 1377,
    'FAILURE': 477,
    'ROOT_CAUSE': 2154,
    'ACTION': 1452,
    'VERIFICATION': 397,
}
EXPECTED_RELATION_TYPE_COUNTS = {
    'CAUSE': 3564,
    'TREAT': 1136,
    'VERIFY': 809,
    'TEMPORAL': 375,
}
FALLBACK_SECTION_TARGET_CHARS = int(os.environ.get('FALLBACK_SECTION_TARGET_CHARS', '320'))
FALLBACK_SECTION_MIN_CHARS = int(os.environ.get('FALLBACK_SECTION_MIN_CHARS', '120'))
FALLBACK_SECTION_MAX_COUNT = int(os.environ.get('FALLBACK_SECTION_MAX_COUNT', '8'))
FALLBACK_SPLIT_CHARS = '\u3002\uff01\uff1f\uff1b!?;\n'
ALLOWED_RELATION_ROLE_TRIPLES = {
    ('ROOT_CAUSE', 'FAILURE', 'CAUSE'),
    ('ROOT_CAUSE', 'PHENOMENON', 'CAUSE'),
    ('FAILURE', 'PHENOMENON', 'CAUSE'),
    ('ACTION', 'PHENOMENON', 'TREAT'),
    ('ACTION', 'FAILURE', 'TREAT'),
    ('ACTION', 'ROOT_CAUSE', 'TREAT'),
    ('VERIFICATION', 'ACTION', 'VERIFY'),
    ('VERIFICATION', 'PHENOMENON', 'VERIFY'),
    ('VERIFICATION', 'FAILURE', 'VERIFY'),
    ('ACTION', 'ACTION', 'TEMPORAL'),
    ('VERIFICATION', 'VERIFICATION', 'TEMPORAL'),
}


def normalize_documents(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        values = list(data.values())
        if values and isinstance(values[0], dict):
            return values
    raise TypeError(f'unsupported annotation container type: {type(data)}')


def validate_manuscript_annotation_statistics(documents: List[Dict]) -> Dict:
    event_counter = Counter()
    relation_counter = Counter()
    invalid_role_triples = []
    for document in documents:
        event_by_id = {
            event.get('event_id'): event
            for event in document.get('event_mentions', [])
        }
        for event in document.get('event_mentions', []):
            event_counter[event.get('type', '')] += 1
        for relation in document.get('relations', []):
            relation_type = relation.get('type', '')
            relation_counter[relation_type] += 1
            head = event_by_id.get(relation.get('head'))
            tail = event_by_id.get(relation.get('tail'))
            role_triple = (
                head.get('type') if head else None,
                tail.get('type') if tail else None,
                relation_type,
            )
            if role_triple not in ALLOWED_RELATION_ROLE_TRIPLES:
                invalid_role_triples.append({
                    'doc_id': str(document.get('doc_id', '')),
                    'relation_id': relation.get('relation_id', ''),
                    'head': relation.get('head', ''),
                    'tail': relation.get('tail', ''),
                    'role_triple': role_triple,
                })

    observed = {
        'document_count': len(documents),
        'event_count': sum(event_counter.values()),
        'relation_count': sum(relation_counter.values()),
        'event_type_counts': dict(sorted(event_counter.items())),
        'relation_type_counts': dict(sorted(relation_counter.items())),
    }
    expected = {
        'document_count': EXPECTED_DOCUMENT_COUNT,
        'event_count': sum(EXPECTED_EVENT_TYPE_COUNTS.values()),
        'relation_count': sum(EXPECTED_RELATION_TYPE_COUNTS.values()),
        'event_type_counts': EXPECTED_EVENT_TYPE_COUNTS,
        'relation_type_counts': EXPECTED_RELATION_TYPE_COUNTS,
    }

    if (
        observed['document_count'] != expected['document_count']
        or observed['event_type_counts'] != expected['event_type_counts']
        or observed['relation_type_counts'] != expected['relation_type_counts']
        or invalid_role_triples
    ):
        raise RuntimeError(
            'Annotation data do not match the manuscript dataset statistics. '
            'Use the complete 286-record gold_sample.jsonl before running the final pipeline. '
            + json.dumps(
                {
                    'expected': expected,
                    'observed': observed,
                    'invalid_role_triples': invalid_role_triples[:20],
                },
                ensure_ascii=False,
            )
        )
    return observed


def normalize_section_name(name: str) -> str:
    mapping = {
        '异常现象描述': 'phenomenon',
        '原因分析': 'cause',
        '排查与处理过程': 'process',
        '后果与损失': 'loss',
        '所需工具': 'tools',
        '备件': 'spares',
    }
    return mapping.get(name, name.strip().lower())


def build_section_records_from_field(document: Dict) -> List[Dict]:
    sections = document.get('sections', {}) or {}
    records = []
    for key, value in sections.items():
        if not isinstance(value, dict):
            continue
        section_name = value.get('name', key)
        section_key = normalize_section_name(section_name)
        start = int(value.get('start', -1))
        end = int(value.get('end', -1))
        text = value.get('text', '')
        if start < 0 or end <= start:
            continue
        records.append({
            'section_key': section_key,
            'section_name': section_name,
            'start': start,
            'end': end,
            'text': text,
            'source': 'field',
        })
    records.sort(key=lambda item: (item['start'], item['end'], item['section_key']))
    return records


def split_long_text_span(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    spans = []
    cursor = start
    while end - cursor > FALLBACK_SECTION_TARGET_CHARS:
        hard_limit = min(end, cursor + FALLBACK_SECTION_TARGET_CHARS)
        soft_min = min(hard_limit, cursor + FALLBACK_SECTION_MIN_CHARS)
        split_at = None
        for idx in range(hard_limit, soft_min - 1, -1):
            if text[idx - 1] in FALLBACK_SPLIT_CHARS:
                split_at = idx
                break
        if split_at is None or split_at <= cursor:
            split_at = hard_limit
        spans.append((cursor, split_at))
        cursor = split_at
    if cursor < end:
        spans.append((cursor, end))
    return spans


def choose_midpoint_split(text: str, start: int, end: int) -> int:
    lower = start + FALLBACK_SECTION_MIN_CHARS
    upper = end - FALLBACK_SECTION_MIN_CHARS
    midpoint = (start + end) // 2
    if lower >= upper:
        return midpoint
    candidates = [
        idx for idx in range(lower, upper + 1)
        if text[idx - 1] in FALLBACK_SPLIT_CHARS
    ]
    if not candidates:
        return midpoint
    return min(candidates, key=lambda idx: abs(idx - midpoint))


def merge_section_spans_to_limit(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    max_count = max(1, FALLBACK_SECTION_MAX_COUNT)
    merged = list(spans)
    while len(merged) > max_count:
        next_round = []
        for idx in range(0, len(merged), 2):
            if idx + 1 < len(merged):
                next_round.append((merged[idx][0], merged[idx + 1][1]))
            else:
                next_round.append(merged[idx])
        merged = next_round
    return merged


def build_section_records_from_text_blocks(document: Dict) -> List[Dict]:
    text = document.get('text', '') or ''
    if not text:
        return []

    unit_spans = []
    for sentence in document.get('sentences', []) or []:
        try:
            start = int(sentence.get('start', -1))
            end = int(sentence.get('end', -1))
        except Exception:
            continue
        if 0 <= start < end <= len(text) and text[start:end].strip():
            unit_spans.append((start, end))

    if not unit_spans:
        unit_spans = [(0, len(text))]

    section_spans = []
    current_start = None
    current_end = None
    for start, end in sorted(unit_spans):
        for piece_start, piece_end in split_long_text_span(text, start, end):
            if current_start is None:
                current_start, current_end = piece_start, piece_end
                continue
            merged_len = piece_end - current_start
            current_len = current_end - current_start
            if current_len >= FALLBACK_SECTION_MIN_CHARS and merged_len > FALLBACK_SECTION_TARGET_CHARS:
                section_spans.append((current_start, current_end))
                current_start, current_end = piece_start, piece_end
            else:
                current_end = piece_end

    if current_start is not None and current_end is not None and current_end > current_start:
        section_spans.append((current_start, current_end))

    if len(section_spans) == 1:
        start, end = section_spans[0]
        if end - start >= 2 * FALLBACK_SECTION_MIN_CHARS:
            split_at = choose_midpoint_split(text, start, end)
            section_spans = [(start, split_at), (split_at, end)]

    section_spans = merge_section_spans_to_limit(section_spans)
    records = []
    for idx, (start, end) in enumerate(section_spans, start=1):
        if end <= start or not text[start:end].strip():
            continue
        records.append({
            'section_key': f'text_block_{idx}',
            'section_name': f'Text block {idx}',
            'start': start,
            'end': end,
            'text': text[start:end],
            'source': 'text_block',
        })
    return records


SECTION_ANCHOR_PATTERNS = [
    ('cause', '原因分析', re.compile(r'(?:故障分析表明|进一步分析表明|通过.*?分析|该故障与|故障与)')),
    ('process', '排查与处理过程', re.compile(r'(?:排查处理中|处理时|经现场检查|随后在|先完成|先调整|需要先)')),
    ('loss', '后果与损失', re.compile(r'(?:此次故障|本次故障|最终造成|导致|造成).{0,30}(?:报废|换新|附加试车|延误|损失|成本|周期)')),
]


def build_section_records_from_text_rules(document: Dict) -> List[Dict]:
    text = document.get('text', '') or ''
    if not text:
        return []

    anchors = []
    for section_key, section_name, pattern in SECTION_ANCHOR_PATTERNS:
        match = pattern.search(text)
        if match:
            anchors.append((match.start(), section_key, section_name))

    deduped = {}
    for start, section_key, section_name in anchors:
        deduped.setdefault(section_key, (start, section_name))
    anchors = sorted((start, key, name) for key, (start, name) in deduped.items())
    if not anchors:
        return []

    boundaries = [(0, 'phenomenon', '异常现象描述')]
    for start, section_key, section_name in anchors:
        if start > 0:
            boundaries.append((start, section_key, section_name))
    boundaries = sorted(dict((start, (key, name)) for start, key, name in boundaries).items())

    records = []
    for idx, (start, (section_key, section_name)) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        if end <= start:
            continue
        records.append({
            'section_key': section_key,
            'section_name': section_name,
            'start': start,
            'end': end,
            'text': text[start:end],
            'source': 'text_rule',
        })
    return records


def build_section_records(document: Dict) -> List[Dict]:
    records = build_section_records_from_field(document)
    if records:
        return records
    records = build_section_records_from_text_rules(document)
    if records:
        return records
    return build_section_records_from_text_blocks(document)


def build_view_records(text: str, section_records: List[Dict]) -> List[Dict]:
    views = [{
        'view_id': 'document',
        'view_type': 'document',
        'source_sections': [item['section_key'] for item in section_records] or ['document'],
        'start': 0,
        'end': len(text),
        'text': text,
    }]

    for section in section_records:
        views.append({
            'view_id': f"section::{section['section_key']}",
            'view_type': 'section',
            'source_sections': [section['section_key']],
            'start': section['start'],
            'end': section['end'],
            'text': text[section['start']:section['end']],
        })

    if len(section_records) >= 2:
        for left, right in zip(section_records[:-1], section_records[1:]):
            start = min(left['start'], right['start'])
            end = max(left['end'], right['end'])
            views.append({
                'view_id': f"bridge::{left['section_key']}+{right['section_key']}",
                'view_type': 'bridge',
                'source_sections': [left['section_key'], right['section_key']],
                'start': start,
                'end': end,
                'text': text[start:end],
            })

    return views



def build_document_level_sample(document: Dict) -> Dict:
    document = ensure_sentences(document)
    text = document.get('text', '')
    section_records = build_section_records(document)
    view_records = build_view_records(text, section_records)
    return {
        'sample_id': f"{document['doc_id']}::document",
        'doc_id': str(document['doc_id']),
        'title': document.get('title', ''),
        'category': document.get('category', ''),
        'sent_id': 0,
        'segment_start': 0,
        'segment_end': len(text),
        'section': 'document',
        'raw_text': text,
        'text': text,
        'event_mentions': document.get('event_mentions', []),
        'relations': document.get('relations', []),
        'sections': section_records,
        'views': view_records,
    }


def keep_negative_samples(samples: List[Dict], rng: random.Random) -> List[Dict]:
    if NEGATIVE_SAMPLE_KEEP_RATIO >= 1.0:
        return samples
    kept = []
    for sample in samples:
        if sample.get('event_mentions'):
            kept.append(sample)
        elif rng.random() < NEGATIVE_SAMPLE_KEEP_RATIO:
            kept.append(sample)
    return kept


def collect_document_statistics(document: Dict) -> Dict:
    event_mentions = document.get('event_mentions', [])
    relations = document.get('relations', [])
    event_types = sorted(set(event['type'] for event in event_mentions))
    relation_types = sorted(set(relation['type'] for relation in relations))
    return {
        'doc_id': str(document['doc_id']),
        'event_types': event_types,
        'relation_types': relation_types,
        'event_count': len(event_mentions),
        'relation_count': len(relations),
        'sentence_count': len(document.get('sentences', [])),
        'text_length': len(document.get('text', '')),
    }


def assign_documents_to_folds(documents: List[Dict], num_folds: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    infos = []
    for document in documents:
        document = ensure_sentences(document)
        info = collect_document_statistics(document)
        info['shuffle_key'] = rng.random()
        infos.append(info)
    infos.sort(key=lambda item: (
        len(item['event_types']),
        len(item['relation_types']),
        item['event_count'],
        item['relation_count'],
        item['shuffle_key'],
    ), reverse=True)

    fold_document_ids = [[] for _ in range(num_folds)]
    fold_event_type_counts = [Counter() for _ in range(num_folds)]
    fold_relation_type_counts = [Counter() for _ in range(num_folds)]
    fold_event_counts = [0 for _ in range(num_folds)]
    fold_relation_counts = [0 for _ in range(num_folds)]

    for info in infos:
        best_fold_index = None
        best_score = None
        for fold_index in range(num_folds):
            event_overlap = sum(fold_event_type_counts[fold_index][t] for t in info['event_types'])
            relation_overlap = sum(fold_relation_type_counts[fold_index][t] for t in info['relation_types'])
            score = (
                event_overlap,
                relation_overlap,
                len(fold_document_ids[fold_index]),
                fold_event_counts[fold_index],
                fold_relation_counts[fold_index],
                fold_index,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_fold_index = fold_index
        fold_document_ids[best_fold_index].append(info['doc_id'])
        fold_event_counts[best_fold_index] += info['event_count']
        fold_relation_counts[best_fold_index] += info['relation_count']
        for event_type in info['event_types']:
            fold_event_type_counts[best_fold_index][event_type] += 1
        for relation_type in info['relation_types']:
            fold_relation_type_counts[best_fold_index][relation_type] += 1

    assignment = {}
    for fold_index, doc_ids in enumerate(fold_document_ids):
        for doc_id in doc_ids:
            assignment[doc_id] = fold_index
    return assignment


def load_existing_fold_assignment(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    assignment = payload.get('fold_assignment', payload)
    return {str(doc_id): int(fold_idx) for doc_id, fold_idx in assignment.items()}


def load_or_build_fold_assignment(documents: List[Dict], num_folds: int, seed: int) -> Dict[str, int]:
    current_doc_ids = sorted(str(doc['doc_id']) for doc in documents)

    if FIX_FOLD_ASSIGNMENT:
        candidate_paths = []
        if FROZEN_FOLD_PATH is not None:
            candidate_paths.append(FROZEN_FOLD_PATH)
        if OUTPUT_FOLD_PATH not in candidate_paths:
            candidate_paths.append(OUTPUT_FOLD_PATH)

        for path in candidate_paths:
            if path.exists():
                saved_assignment = load_existing_fold_assignment(path)
                saved_doc_ids = sorted(saved_assignment.keys())

                if STRICT_DOC_ID_CHECK and saved_doc_ids != current_doc_ids:
                    missing_in_saved = [doc_id for doc_id in current_doc_ids if doc_id not in saved_assignment]
                    missing_in_current = [doc_id for doc_id in saved_doc_ids if doc_id not in current_doc_ids]
                    raise RuntimeError(
                        '检测到当前文档集合与冻结的 fold_assignment 不一致。'
                        f' 当前 gold 中缺少的 doc_id: {missing_in_current}；'
                        f' 冻结划分中缺少的 doc_id: {missing_in_saved}。'
                        ' 如果你是有意增删文档，请先关闭 FIX_FOLD_ASSIGNMENT，或更新冻结划分文件。'
                    )

                reused = {doc_id: saved_assignment[doc_id] for doc_id in current_doc_ids if doc_id in saved_assignment}
                if len(reused) != len(current_doc_ids):
                    raise RuntimeError('fold_assignment 复用失败，存在未覆盖的 doc_id。')
                print(f'复用已有 fold_assignment: {path}')
                return reused

    print('未找到可复用的 fold_assignment，按当前 gold 重新分配 folds。')
    return assign_documents_to_folds(documents, num_folds, seed)


def summarize_fold_distribution(documents: List[Dict], fold_assignment: Dict[str, int], num_folds: int) -> Dict:
    fold_summary = {}
    for fold_index in range(num_folds):
        fold_documents = [doc for doc in documents if fold_assignment[str(doc['doc_id'])] == fold_index]
        event_counter = Counter()
        relation_counter = Counter()
        total_events = total_relations = total_text_length = total_sentence_count = 0
        for document in fold_documents:
            total_text_length += len(document.get('text', ''))
            total_sentence_count += len(document.get('sentences', []))
            for event in document.get('event_mentions', []):
                event_counter[event['type']] += 1
                total_events += 1
            for relation in document.get('relations', []):
                relation_counter[relation['type']] += 1
                total_relations += 1
        fold_summary[str(fold_index)] = {
            'document_ids': sorted(str(doc['doc_id']) for doc in fold_documents),
            'document_count': len(fold_documents),
            'event_count': total_events,
            'relation_count': total_relations,
            'avg_text_length': total_text_length / len(fold_documents) if fold_documents else 0.0,
            'avg_sentence_count': total_sentence_count / len(fold_documents) if fold_documents else 0.0,
            'event_type_distribution': dict(sorted(event_counter.items())),
            'relation_type_distribution': dict(sorted(relation_counter.items())),
        }
    return fold_summary


def summarize_sample_distribution(samples: List[Dict], fold_assignment: Dict[str, int], num_folds: int) -> Dict:
    summary = {}
    for fold_index in range(num_folds):
        fold_samples = [sample for sample in samples if fold_assignment[sample['doc_id']] == fold_index]
        positive_sample_count = sum(1 for sample in fold_samples if sample.get('event_mentions'))
        negative_sample_count = len(fold_samples) - positive_sample_count
        event_counter = Counter()
        relation_counter = Counter()
        total_text_length = total_view_count = total_section_count = 0
        section_key_counter = Counter()
        view_type_counter = Counter()
        section_source_counter = Counter()
        for sample in fold_samples:
            total_text_length += len(sample.get('text', ''))
            total_view_count += len(sample.get('views', []))
            total_section_count += len(sample.get('sections', []))
            for section in sample.get('sections', []):
                section_key_counter[section['section_key']] += 1
                section_source_counter[section.get('source', 'unknown')] += 1
            for view in sample.get('views', []):
                view_type_counter[view['view_type']] += 1
            for event in sample.get('event_mentions', []):
                event_counter[event['type']] += 1
            for relation in sample.get('relations', []):
                relation_counter[relation['type']] += 1
        summary[str(fold_index)] = {
            'sample_count': len(fold_samples),
            'positive_sample_count': positive_sample_count,
            'negative_sample_count': negative_sample_count,
            'avg_text_length': total_text_length / len(fold_samples) if fold_samples else 0.0,
            'avg_view_count': total_view_count / len(fold_samples) if fold_samples else 0.0,
            'avg_section_count': total_section_count / len(fold_samples) if fold_samples else 0.0,
            'section_distribution': dict(sorted(section_key_counter.items())),
            'section_source_distribution': dict(sorted(section_source_counter.items())),
            'view_type_distribution': dict(sorted(view_type_counter.items())),
            'event_type_distribution': dict(sorted(event_counter.items())),
            'relation_type_distribution': dict(sorted(relation_counter.items())),
        }
    return summary


def validate_samples(samples: List[Dict]) -> Dict:
    total_view_count = sum(len(sample.get('views', [])) for sample in samples)
    total_section_count = sum(len(sample.get('sections', [])) for sample in samples)
    avg_view_count = total_view_count / len(samples) if samples else 0.0
    avg_section_count = total_section_count / len(samples) if samples else 0.0
    if samples and avg_view_count < 1.0:
        raise RuntimeError(f'02 生成失败：avg_view_count={avg_view_count:.3f}，没有写入全文视图')
    if samples and avg_section_count < 1.0:
        raise RuntimeError(f'02 generated no section views: avg_section_count={avg_section_count:.3f}')
    return {'avg_view_count': avg_view_count, 'avg_section_count': avg_section_count}


def main():
    rng = random.Random(RANDOM_SEED)
    documents = normalize_documents(load_jsonl(str(INPUT_ANNOTATION_PATH)))
    processed_documents = [ensure_sentences(document) for document in documents]
    manuscript_dataset_stats = (
        validate_manuscript_annotation_statistics(processed_documents)
        if VALIDATE_MANUSCRIPT_DATASET
        else {'validation_disabled': True}
    )
    fold_assignment = load_or_build_fold_assignment(processed_documents, NUM_FOLDS, RANDOM_SEED)
    all_samples = [build_document_level_sample(document) for document in processed_documents]
    all_samples = keep_negative_samples(all_samples, rng)
    validation_stats = validate_samples(all_samples)

    OUTPUT_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_samples, str(OUTPUT_SAMPLE_PATH))
    fold_payload = {'num_folds': NUM_FOLDS, 'seed': RANDOM_SEED, 'fold_assignment': fold_assignment}
    save_json(fold_payload, str(OUTPUT_FOLD_PATH))

    # Keep a frozen split file so later runs reuse the same outer folds.
    if FIX_FOLD_ASSIGNMENT and FROZEN_FOLD_PATH is not None:
        FROZEN_FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_json(fold_payload, str(FROZEN_FOLD_PATH))
    summary_payload = {
        'num_documents': len(processed_documents),
        'num_samples': len(all_samples),
        'manuscript_dataset_stats': manuscript_dataset_stats,
        'validation_stats': validation_stats,
        'task_mode': 'document_level_event_samples_with_text_block_fallback_sections',
        'fold_document_summary': summarize_fold_distribution(processed_documents, fold_assignment, NUM_FOLDS),
        'fold_sample_summary': summarize_sample_distribution(all_samples, fold_assignment, NUM_FOLDS),
    }
    save_json(summary_payload, str(OUTPUT_SUMMARY_PATH))
    print(f'saved document-level samples to: {OUTPUT_SAMPLE_PATH}')
    print(f'saved fold assignment to: {OUTPUT_FOLD_PATH}')
    if FIX_FOLD_ASSIGNMENT and FROZEN_FOLD_PATH is not None:
        print(f'saved frozen fold assignment to: {FROZEN_FOLD_PATH}')
    print(f'saved fold summary to: {OUTPUT_SUMMARY_PATH}')
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
