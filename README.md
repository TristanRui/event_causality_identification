# Event Causality Identification

This repository contains the final five-fold event causality identification pipeline for DC-MCPG extraction.

## Data Availability

The maintenance records and gold annotations used in the manuscript contain aviation engine domain knowledge and cannot be redistributed. The repository therefore keeps only empty placeholder files under `data/raw/` and `data/annotations/`.

To support independent reuse, the expected input schema is documented below. The placeholders are format descriptions only. They are not samples from the manuscript dataset and are not used for the reported results.

## Contents

- `data/raw/`: local raw input text directory. Confidential data are not included in this repository.
- `data/annotations/`: local gold annotation directory. Confidential gold files are not included in this repository.
- `scripts/01_prepare_raw_docs.py`: build raw document records by reading one complete document per `.txt` file from `data/raw/`.
- `scripts/02_prepare_event_folds.py`: build document-level event samples and five-fold splits.
- `scripts/03_train_event_extractor.py`: train the event extractor for all five folds.
- `scripts/04_prepare_relation_graph_samples.py`: build relation graph samples from event predictions.
- `scripts/05_train_relation_graph.py`: train the relation graph model for all five folds.
- `scripts/06_decode_relation_graph.py`: decode relation graphs for all five folds.
- `scripts/07_eval_process_level_utility.py`: evaluate process-level utility.
- `scripts/run_pipeline.py`: run the full five-fold pipeline.

## Input Format

Raw input is document-level: place one complete maintenance record in each `.txt` file under `data/raw/`. The raw preparation script reads every non-empty `.txt` file as one document and does not split documents by blank lines or project-specific markers.

For training and evaluation, provide `data/annotations/gold_sample.jsonl`. Each line is one JSON object with at least:

- `doc_id`: document identifier.
- `title`: optional document title.
- `category`: optional document category.
- `text`: the full document text.
- `event_mentions`: character-level event spans.
- `relations`: directed typed edges between event IDs.
- `sections`: optional section spans. If omitted, the code builds text-block section and bridge views automatically.

Event spans use zero-based, end-exclusive character offsets into `text`. Event types must be one of `PHENOMENON`, `FAILURE`, `ROOT_CAUSE`, `ACTION`, or `VERIFICATION`. Relation types must be one of `CAUSE`, `TEMPORAL`, `TREAT`, or `VERIFY`.

Raw files should contain complete document-level records. Each document should include the information needed to identify:

- abnormal phenomena observed in the record;
- failure states, if explicitly described;
- root causes or diagnostic conclusions;
- corrective actions;
- verification evidence after treatment.

The gold annotation file should use the following shape. The `text` value is the same complete document text as the corresponding raw file. Offsets are zero-based and end-exclusive character offsets into `text`.

```jsonc
{
  "doc_id": "<document_id>",
  "title": "<optional_title>",
  "category": "<optional_category>",
  "text": "<complete_document_text>",
  "event_mentions": [
    {"event_id": "E1", "type": "PHENOMENON", "start": <start_offset>, "end": <end_offset>, "text": "<phenomenon_span_text>"},
    {"event_id": "E2", "type": "FAILURE", "start": <start_offset>, "end": <end_offset>, "text": "<failure_span_text>"},
    {"event_id": "E3", "type": "ROOT_CAUSE", "start": <start_offset>, "end": <end_offset>, "text": "<root_cause_span_text>"},
    {"event_id": "E4", "type": "ACTION", "start": <start_offset>, "end": <end_offset>, "text": "<action_span_text>"},
    {"event_id": "E5", "type": "VERIFICATION", "start": <start_offset>, "end": <end_offset>, "text": "<verification_span_text>"}
  ],
  "relations": [
    {"relation_id": "R1", "head": "<ROOT_CAUSE_EVENT_ID>", "tail": "<FAILURE_OR_PHENOMENON_EVENT_ID>", "type": "CAUSE"},
    {"relation_id": "R2", "head": "<FAILURE_EVENT_ID>", "tail": "<PHENOMENON_EVENT_ID>", "type": "CAUSE"},
    {"relation_id": "R3", "head": "<ACTION_EVENT_ID>", "tail": "<PHENOMENON_OR_FAILURE_OR_ROOT_CAUSE_EVENT_ID>", "type": "TREAT"},
    {"relation_id": "R4", "head": "<VERIFICATION_EVENT_ID>", "tail": "<ACTION_OR_PHENOMENON_OR_FAILURE_EVENT_ID>", "type": "VERIFY"},
    {"relation_id": "R5", "head": "<ACTION_EVENT_ID>", "tail": "<ACTION_EVENT_ID>", "type": "TEMPORAL"},
    {"relation_id": "R6", "head": "<VERIFICATION_EVENT_ID>", "tail": "<VERIFICATION_EVENT_ID>", "type": "TEMPORAL"}
  ],
  "sections": {
    "<optional_section_key>": {
      "name": "<section_name>",
      "start": <start_offset>,
      "end": <end_offset>,
      "text": "<section_text>"
    }
  }
}
```

The relation directions must follow the manuscript schema:

- `CAUSE`: `ROOT_CAUSE -> FAILURE`, `ROOT_CAUSE -> PHENOMENON`, or `FAILURE -> PHENOMENON`.
- `TREAT`: `ACTION -> PHENOMENON`, `ACTION -> FAILURE`, or `ACTION -> ROOT_CAUSE`.
- `VERIFY`: `VERIFICATION -> ACTION`, `VERIFICATION -> PHENOMENON`, or `VERIFICATION -> FAILURE`.
- `TEMPORAL`: `ACTION -> ACTION` or `VERIFICATION -> VERIFICATION`.

The final manuscript pipeline validates the private dataset statistics by default: 286 documents, 5857 event nodes, and 5884 relation edges with the published type distributions. For format checks on another private dataset, set `VALIDATE_MANUSCRIPT_DATASET=0`; a small placeholder-style file is not sufficient for five-fold model training.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full five-fold pipeline:

```bash
python scripts/run_pipeline.py
```

Before running the pipeline, place the required private data files locally:

- private raw document `.txt` files under `data/raw/`; each file is treated as one document-level input;
- `data/annotations/gold_sample.jsonl`

Keep the tracked `data/raw/case_input.txt` placeholder empty. Use separate private file names for real raw documents so they stay ignored by Git.

The training scripts load `hfl/chinese-roberta-wwm-ext` from the local Hugging Face cache by default. Download that public model into the local cache before running, or set `PRETRAINED_MODEL_NAME` to a local model directory.

The full pipeline starts with raw document preparation and five-fold sample preparation. The raw preparation stage does not split by blank lines or project-specific case markers; it reads every non-empty `.txt` file under `data/raw/` as one document. To run only selected stages, pass `--stages`, for example:

```bash
python scripts/run_pipeline.py --stages raw,folds,event
```

By default, generated files are written under `outputs/five_fold_run/` and `data/processed/`.

## Protocol

The default pipeline follows the manuscript protocol:

- five outer document-level folds;
- for each outer run, the next fold is used as dev and the remaining three folds as train;
- event observations for outer train documents are generated by inner out-of-fold event extractors within the outer training documents;
- event observations for outer dev/test documents are generated by an event extractor trained only on the outer train folds;
- relation graph training uses predicted event observations, not gold event nodes.

`scripts/02_prepare_event_folds.py` validates the annotation statistics against the manuscript dataset by default: 286 documents, 5857 event nodes, and 5884 relation edges with the published type distributions. These data must be supplied locally and should not be committed.
