from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
FOLDS = "0,1,2,3,4"
FOLD_LIST = [0, 1, 2, 3, 4]
STAGES = ("raw", "folds", "event", "graph", "relation", "decode", "utility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-fold event causality identification pipeline."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "five_fold_run",
        help="Root directory for a new five-fold experiment.",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated subset of: raw,folds,event,graph,relation,decode,utility; default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without running them.",
    )
    return parser.parse_args()


def resolve_stages(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(STAGES)
    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(STAGES))
    if invalid:
        raise ValueError(f"Unknown stages: {', '.join(invalid)}")
    if not selected:
        raise ValueError("At least one stage must be selected.")
    return selected


def base_environment(output_root: Path) -> dict[str, str]:
    return {
        "RUN_FOLDS": FOLDS,
        "NUM_FOLDS": "5",
        "EVENT_OUTPUT_DIR": str(output_root / "event_extraction"),
        "EVENT_RUN_DIR_NAME": "event_extraction",
        "PREDICTION_ROOT": str(output_root / "event_extraction"),
        "OUTPUT_GRAPH_SAMPLE_ROOT": str(output_root / "graph_samples"),
        "INPUT_GRAPH_SAMPLE_ROOT": str(output_root / "graph_samples"),
        "SAMPLE_LOAD_MODE": "direct_split",
        "NODE_SOURCE": "predicted_events",
        "TRAIN_TARGET_KEY": "train_strict_exact_unmatched_ignore",
        "EARLY_STOP_TARGET_KEY": "strict_exact",
        "TEST_TARGET_KEYS": "strict_exact",
        "CHECKPOINT_NODE_SOURCE": "predicted_events",
        "CHECKPOINT_TRAIN_TARGET_KEY": "train_strict_exact_unmatched_ignore",
        "CHECKPOINT_ROOT": str(output_root / "relation_model"),
        "DECODE_OUTPUT_DIR": str(output_root / "graph_predictions"),
        "DATASET_TARGET_KEY": "strict_exact",
        "FORCE_EVAL_TARGET_KEYS": "strict_exact",
    }


def validate_json(path: Path, predicate: Callable[[dict], bool], stage: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{stage} did not create {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not predicate(payload):
        raise RuntimeError(f"{stage} output is not a complete five-fold result: {path}")


def run_stage(name: str, command: list[str], env_overrides: dict[str, str], dry_run: bool) -> None:
    print(f"\n===== {name} =====", flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(env_overrides)
    subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    selected = resolve_stages(args.stages)
    env = base_environment(output_root)

    manifest = {
        "python": sys.executable,
        "folds": FOLD_LIST,
        "output_root": str(output_root),
        "stages": selected,
        "environment": env,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "pipeline_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if "raw" in selected:
        run_stage("raw document preparation", [sys.executable, "scripts/01_prepare_raw_docs.py"], env, args.dry_run)
        if not args.dry_run:
            raw_docs_path = PROJECT_DIR / "data" / "processed" / "raw_docs.jsonl"
            if not raw_docs_path.exists():
                raise RuntimeError(f"raw document preparation did not create {raw_docs_path}")

    if "folds" in selected:
        run_stage("five-fold sample preparation", [sys.executable, "scripts/02_prepare_event_folds.py"], env, args.dry_run)
        if not args.dry_run:
            validate_json(
                PROJECT_DIR / "data" / "processed" / "event_document_folds.json",
                lambda item: item.get("num_folds") == 5
                and set(item.get("fold_assignment", {}).values()) == set(range(5)),
                "five-fold sample preparation",
            )

    if "event" in selected:
        run_stage("event extraction", [sys.executable, "scripts/03_train_event_extractor.py"], env, args.dry_run)
        if not args.dry_run:
            validate_json(
                output_root / "event_extraction" / "cross_validation_summary.json",
                lambda item: item.get("evaluation_mode") == "five_fold_cross_validation" and len(item.get("fold_results", [])) == 5,
                "event extraction",
            )

    if "graph" in selected:
        run_stage("graph sample preparation", [sys.executable, "scripts/04_prepare_relation_graph_samples.py"], env, args.dry_run)
        if not args.dry_run:
            validate_json(
                output_root / "graph_samples" / "relation_graph_predicted_summary.json",
                lambda item: all(
                    {
                        entry.get("fold_index")
                        for entry in item.get("file_summaries", [])
                        if entry.get("split") == split_name
                    } == set(range(5))
                    for split_name in ["train", "dev", "test"]
                ),
                "graph sample preparation",
            )

    if "relation" in selected:
        run_stage("relation graph training", [sys.executable, "scripts/05_train_relation_graph.py"], env, args.dry_run)
        if not args.dry_run:
            validate_json(
                output_root / "relation_model" / "cross_validation_summary.json",
                lambda item: item.get("run_folds") == FOLD_LIST and len(item.get("fold_results", [])) == 5,
                "relation graph training",
            )

    if "decode" in selected:
        run_stage("end-to-end graph decoding", [sys.executable, "scripts/06_decode_relation_graph.py"], env, args.dry_run)
        if not args.dry_run:
            validate_json(
                output_root / "graph_predictions" / "decode_summary.json",
                lambda item: item.get("run_folds") == FOLD_LIST and len(item.get("decode_summaries", [])) == 5,
                "end-to-end graph decoding",
            )

    if "utility" in selected:
        run_stage(
            "process utility evaluation",
            [
                sys.executable,
                "scripts/07_eval_process_level_utility.py",
                "--prediction-root",
                str(output_root / "graph_predictions"),
                "--output-dir",
                str(output_root / "process_utility"),
            ],
            env,
            args.dry_run,
        )
        if not args.dry_run:
            validate_json(
                output_root / "process_utility" / "process_level_metrics.json",
                lambda item: bool(item.get("metrics")),
                "process utility evaluation",
            )


if __name__ == "__main__":
    main()
