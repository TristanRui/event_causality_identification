from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from config import SCHEMA
from utils.io_utils import save_jsonl
from utils.text_utils import normalize_text, split_sentences


INPUT_RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_JSONL = PROJECT_DIR / "data" / "processed" / "raw_docs.jsonl"
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")


def read_text_with_fallback(path: Path) -> str:
    last_error = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError(f"Unable to read text file: {path}")
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Unable to decode {path} with {TEXT_ENCODINGS}: {last_error.reason}",
    )


def build_doc_id(path: Path, text: str) -> str:
    stem = path.stem.strip().replace(" ", "_")
    digest = hashlib.sha1(f"{path.name}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def build_document_record(path: Path, text: str) -> Dict:
    title = path.stem.strip()
    doc_id = build_doc_id(path, text)
    return {
        "doc_id": doc_id,
        "case_no": "",
        "title": title,
        "category": "",
        "text": text,
        "sentences": split_sentences(text),
        "schema": SCHEMA,
        "event_mentions": [],
        "relations": [],
        "source_file": path.name,
    }


def load_document_texts(input_dir: Path) -> List[Dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Raw text directory does not exist: {input_dir}")

    docs: List[Dict] = []
    for path in sorted(input_dir.glob("*.txt")):
        if not path.is_file():
            continue
        text = normalize_text(read_text_with_fallback(path)).strip()
        if not text:
            continue
        docs.append(build_document_record(path, text))

    doc_ids = [doc["doc_id"] for doc in docs]
    if len(doc_ids) != len(set(doc_ids)):
        duplicates = sorted({doc_id for doc_id in doc_ids if doc_ids.count(doc_id) > 1})
        raise RuntimeError(f"Duplicate document ids generated from input files: {duplicates[:20]}")

    return docs


def main() -> None:
    docs = load_document_texts(INPUT_RAW_DIR)
    if not docs:
        raise RuntimeError(
            f"No non-empty .txt documents found under {INPUT_RAW_DIR}. "
            "Place one complete document per .txt file before running this stage."
        )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(docs, str(OUTPUT_JSONL))

    print(f"Loaded {len(docs)} document-level text files from: {INPUT_RAW_DIR}")
    print(f"Saved raw document records to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
