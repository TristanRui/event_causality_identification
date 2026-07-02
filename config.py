from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANN_DIR = DATA_DIR / "annotations"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs"

RANDOM_SEED = 42
FOLD_COUNT = 5

EVENT_TYPES = [
    "PHENOMENON",
    "FAILURE",
    "ROOT_CAUSE",
    "ACTION",
    "VERIFICATION",
]

RELATION_TYPES = [
    "CAUSE",
    "TEMPORAL",
    "TREAT",
    "VERIFY",
]

EVENT_BIO_LABELS = ["O"]
for event_type in EVENT_TYPES:
    EVENT_BIO_LABELS.append(f"B-{event_type}")
    EVENT_BIO_LABELS.append(f"I-{event_type}")

EVENT_LABEL2ID = {label: idx for idx, label in enumerate(EVENT_BIO_LABELS)}
EVENT_ID2LABEL = {idx: label for label, idx in EVENT_LABEL2ID.items()}

REL_LABELS = ["NONE"] + RELATION_TYPES
REL_LABEL2ID = {label: idx for idx, label in enumerate(REL_LABELS)}
REL_ID2LABEL = {idx: label for label, idx in REL_LABEL2ID.items()}

EVENT_TYPE2ID = {event_type: idx for idx, event_type in enumerate(EVENT_TYPES)}
EVENT_ID2TYPE = {idx: event_type for event_type, idx in EVENT_TYPE2ID.items()}

SCHEMA = {
    "event_types": EVENT_TYPES,
    "relation_types": RELATION_TYPES,
}
