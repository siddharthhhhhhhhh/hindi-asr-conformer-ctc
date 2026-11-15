"""Build Kaggle-ready JSONL manifests for the 200K Hindi ASR stage.

Expected Kaggle inputs
----------------------
1. ``hindi-asr-100k-private``: the original 100K audio, validation audio,
   vocabulary, and validation manifest.
2. ``hindi-asr-extra-100k-private``: the additional 100K audio and
   ``train_200k_ctc_manifest.csv``.

The script never copies the 200K recordings into ``/kaggle/working``. It
indexes the read-only FLAC files under ``/kaggle/input`` and writes compact
JSONL manifests containing their real paths. This saves roughly 20 GB of
working-disk space.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_100K_ROOT = Path(
    os.environ.get(
        "HINDI_ASR_100K_ROOT",
        "/kaggle/input/datasets/siddharthroyjj/hindi-asr-100k-private",
    )
)
EXTRA_100K_ROOT = Path(
    os.environ.get(
        "HINDI_ASR_EXTRA_100K_ROOT",
        "/kaggle/input/datasets/siddharthroyjj/hindi-asr-extra-100k-private",
    )
)
WORK_ROOT = Path(
    os.environ.get(
        "HINDI_ASR_200K_WORK_ROOT",
        "/kaggle/working/hindi_asr_200k",
    )
)

COMBINED_CSV = EXTRA_100K_ROOT / "train_200k_ctc_manifest.csv"
SOURCE_VALIDATION_JSON = BASE_100K_ROOT / "validation_manifest.json"
TRAIN_JSON = WORK_ROOT / "train_manifest.json"
VALIDATION_JSON = WORK_ROOT / "validation_manifest.json"


def load_json_lines(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json_lines(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def choose_text(row: dict) -> str:
    for key in ("model_text", "text", "normalized"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("A manifest row has no usable transcript.")


def index_audio_files() -> dict[str, Path]:
    audio_by_name: dict[str, Path] = {}
    duplicates: list[str] = []

    for root in (BASE_100K_ROOT, EXTRA_100K_ROOT):
        if not root.exists():
            raise FileNotFoundError(f"Missing Kaggle input root: {root}")

        for path in root.rglob("*.flac"):
            previous = audio_by_name.get(path.name)
            if previous is not None and previous != path:
                duplicates.append(path.name)
            else:
                audio_by_name[path.name] = path

    if duplicates:
        raise ValueError(
            "Duplicate FLAC basenames prevent safe matching: "
            + str(sorted(set(duplicates))[:10])
        )

    return audio_by_name


def expected_audio_name(row: dict) -> str:
    for key in ("audio_filepath", "audio_path"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return Path(str(value)).name

    if row.get("original_row") is None:
        raise ValueError(
            "A row has neither an audio path nor original_row."
        )

    return f"clip_{int(row['original_row']):07d}.flac"


def standardize_record(row: dict, audio_by_name: dict[str, Path]) -> dict:
    filename = expected_audio_name(row)
    audio_path = audio_by_name.get(filename)
    if audio_path is None:
        raise FileNotFoundError(f"Audio not found for {filename}")

    record = {
        "audio_filepath": str(audio_path),
        "text": choose_text(row),
        "speaker_id": str(row.get("speaker_id", "")),
        "duration": float(row["duration"]),
    }

    if row.get("original_row") is not None:
        record["original_row"] = int(row["original_row"])

    return record


def load_optional_speaker_csv(filename: str) -> set[str]:
    candidates = list(BASE_100K_ROOT.rglob(filename))
    candidates.extend(EXTRA_100K_ROOT.rglob(filename))
    if not candidates:
        return set()

    frame = pd.read_csv(candidates[0])
    if "speaker_id" not in frame.columns:
        return set()
    return set(frame["speaker_id"].dropna().astype(str))


def prepare_manifests() -> tuple[Path, Path]:
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Missing combined 200K manifest: {COMBINED_CSV}"
        )

    validation_source = first_existing(SOURCE_VALIDATION_JSON)
    if validation_source is None:
        raise FileNotFoundError(
            f"Missing validation manifest: {SOURCE_VALIDATION_JSON}"
        )

    print("Indexing the two attached Kaggle audio datasets...")
    audio_by_name = index_audio_files()
    combined = pd.read_csv(COMBINED_CSV)

    if len(combined) != 200_000:
        raise ValueError(
            f"Expected 200000 training rows, found {len(combined)}."
        )
    if "original_row" not in combined.columns:
        raise ValueError("The combined manifest needs original_row.")
    if combined["original_row"].duplicated().any():
        raise ValueError("Duplicate original_row values found in training.")
    if combined["speaker_id"].isna().any():
        raise ValueError("Missing training speaker IDs found.")

    train_speakers = set(combined["speaker_id"].astype(str))
    validation_source_records = load_json_lines(validation_source)
    validation_speakers = {
        str(record.get("speaker_id", ""))
        for record in validation_source_records
        if record.get("speaker_id") is not None
    }
    test_speakers = load_optional_speaker_csv(
        "test_10h_ctc_manifest.csv"
    )

    train_validation_overlap = train_speakers & validation_speakers
    train_test_overlap = train_speakers & test_speakers
    if train_validation_overlap:
        raise ValueError(
            "Train-validation speaker overlap: "
            f"{len(train_validation_overlap)}"
        )
    if train_test_overlap:
        raise ValueError(
            f"Train-test speaker overlap: {len(train_test_overlap)}"
        )

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    train_records = [
        standardize_record(row, audio_by_name)
        for row in combined.to_dict(orient="records")
    ]
    validation_records = [
        standardize_record(record, audio_by_name)
        for record in validation_source_records
    ]

    if len(validation_records) != 567:
        raise ValueError(
            "Expected 567 validation rows, found "
            f"{len(validation_records)}."
        )

    write_json_lines(TRAIN_JSON, train_records)
    write_json_lines(VALIDATION_JSON, validation_records)

    training_hours = sum(
        record["duration"] for record in train_records
    ) / 3600

    print("\n200K MANIFESTS READY")
    print("Indexed FLAC files:", len(audio_by_name))
    print("Training recordings:", len(train_records))
    print("Training audio hours:", round(training_hours, 2))
    print("Training speakers:", len(train_speakers))
    print("Validation recordings:", len(validation_records))
    print("Train-validation speaker overlap: 0")
    if test_speakers:
        print("Train-test speaker overlap: 0")
    else:
        print("Train-test overlap: not rechecked (test CSV not attached)")
    print("Training manifest:", TRAIN_JSON)
    print("Validation manifest:", VALIDATION_JSON)

    return TRAIN_JSON, VALIDATION_JSON


if __name__ == "__main__":
    prepare_manifests()
