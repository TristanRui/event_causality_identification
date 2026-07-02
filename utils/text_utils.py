import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[Dict]:
    sentences = []
    start = 0

    for i, ch in enumerate(text):
        if ch in "。！？；!?;\n":
            sent = text[start:i + 1].strip()
            if sent:
                real_start = text.find(sent, start)
                real_end = real_start + len(sent)
                sentences.append({
                    "sent_id": len(sentences),
                    "text": sent,
                    "start": real_start,
                    "end": real_end
                })
            start = i + 1

    tail = text[start:].strip()
    if tail:
        real_start = text.find(tail, start)
        real_end = real_start + len(tail)
        sentences.append({
            "sent_id": len(sentences),
            "text": tail,
            "start": real_start,
            "end": real_end
        })

    return sentences


def ensure_sentences(doc: Dict) -> Dict:
    if "sentences" not in doc or not doc["sentences"]:
        doc["sentences"] = split_sentences(doc["text"])
    return doc


def find_sentence_id(sentences: List[Dict], start_offset: int, end_offset: int) -> int:
    for sent in sentences:
        if start_offset >= sent["start"] and end_offset <= sent["end"]:
            return sent["sent_id"]
    return -1
