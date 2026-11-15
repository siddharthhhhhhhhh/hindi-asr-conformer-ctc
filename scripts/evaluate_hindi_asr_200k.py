#!/usr/bin/env python3
"""Evaluate the 200K Hindi Conformer-CTC checkpoint on the fixed 10-hour test set."""

from __future__ import annotations

import csv
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.models import Conformer


DOWNLOADS = Path.home() / "Downloads"
WORK_DIR = Path.home() / "Hindi_ASR_Final_Evaluation"
TEST_ZIP = DOWNLOADS / "hindi_asr_test_10h_parts-20261109T163236Z-1-001.zip"
CHECKPOINT_ZIP = DOWNLOADS / "hindi_asr_200k_epoch3_bundle.zip"
VOCAB_PATH = DOWNLOADS / "ctc_vocabulary.json"

TEST_PACKAGE_DIR = WORK_DIR / "test_package"
TEST_ROOT = WORK_DIR / "hindi_asr_test_10h"
TEST_AUDIO_DIR = TEST_ROOT / "audio"
TEST_MANIFEST = TEST_PACKAGE_DIR / "hindi_asr_test_10h_parts" / "test_manifest.json"
CHECKPOINT_DIR = WORK_DIR / "checkpoint"
CHECKPOINT_PATH = CHECKPOINT_DIR / "conformer_200k_best_cer.pt"

PREDICTIONS_PATH = WORK_DIR / "test_200k_predictions.csv"
RESULTS_PATH = WORK_DIR / "test_200k_results.json"

BATCH_SIZE = 4
EXPECTED_RECORDINGS = 5897
EXPECTED_HOURS = 9.495


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        archive.extractall(destination)


def prepare_files() -> None:
    for path in (TEST_ZIP, CHECKPOINT_ZIP, VOCAB_PATH):
        require_file(path)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_MANIFEST.is_file():
        print("Extracting the downloaded test package...")
        TEST_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(TEST_ZIP) as archive:
            archive.extractall(TEST_PACKAGE_DIR)

    audio_count = len(list(TEST_AUDIO_DIR.glob("*.flac")))
    if audio_count != EXPECTED_RECORDINGS:
        print("Extracting the 15 test-audio archives...")
        tar_files = sorted(TEST_MANIFEST.parent.glob("test_audio_part_*.tar"))
        if len(tar_files) != 15:
            raise ValueError(f"Expected 15 test TAR files, found {len(tar_files)}")
        for index, tar_path in enumerate(tar_files, start=1):
            safe_extract_tar(tar_path, WORK_DIR)
            print(f"  extracted {index}/15")

    audio_count = len(list(TEST_AUDIO_DIR.glob("*.flac")))
    if audio_count != EXPECTED_RECORDINGS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDINGS} test recordings, found {audio_count}"
        )

    if not CHECKPOINT_PATH.is_file():
        print("Extracting the epoch-3 checkpoint...")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CHECKPOINT_ZIP) as archive:
            archive.extractall(CHECKPOINT_DIR)

    print(f"Test recordings ready: {audio_count}")


with VOCAB_PATH.open("r", encoding="utf-8") as file:
    vocabulary_data = json.load(file)

CHAR_TO_ID = {
    character: index + 1
    for index, character in enumerate(vocabulary_data["characters"])
}
BLANK_ID = int(vocabulary_data.get("blank_id", 0))
ID_TO_CHAR = {token_id: character for character, token_id in CHAR_TO_ID.items()}
NUMBER_OF_CLASSES = max(max(CHAR_TO_ID.values()), BLANK_ID) + 1


class HindiTestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        with manifest_path.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]
        if len(self.records) != EXPECTED_RECORDINGS:
            raise ValueError(
                f"Expected {EXPECTED_RECORDINGS} manifest rows, found {len(self.records)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        audio_path = TEST_AUDIO_DIR / Path(record["audio_filepath"]).name
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if sample_rate != 16000:
            raise ValueError(f"Unexpected sample rate {sample_rate}: {audio_path}")
        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        text = record["text"]
        target = torch.tensor([CHAR_TO_ID[c] for c in text], dtype=torch.long)
        return waveform, target, text, str(audio_path), float(record["duration"])


MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    power=2.0,
)


def collate_batch(batch):
    waveforms, targets, texts, paths, durations = zip(*batch)
    waveform_lengths = torch.tensor([len(x) for x in waveforms], dtype=torch.long)
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    features = MEL_TRANSFORM(padded_waveforms)
    features = torch.log(features.clamp_min(1e-10)).transpose(1, 2)
    feature_lengths = waveform_lengths // 160 + 1

    for index, length in enumerate(feature_lengths):
        valid = features[index, : int(length)]
        features[index, : int(length)] = (
            valid - valid.mean(dim=0, keepdim=True)
        ) / valid.std(dim=0, keepdim=True).clamp_min(1e-5)

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "targets": pad_sequence(targets, batch_first=True, padding_value=BLANK_ID),
        "target_lengths": torch.tensor([len(x) for x in targets], dtype=torch.long),
        "texts": texts,
        "paths": paths,
        "durations": durations,
    }


class ConvSubsampling(nn.Module):
    def __init__(self, input_features: int = 80, model_dimension: int = 144):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, model_dimension, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                model_dimension,
                model_dimension,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )
        reduced_frequency = (input_features + 1) // 2
        reduced_frequency = (reduced_frequency + 1) // 2
        self.projection = nn.Linear(
            model_dimension * reduced_frequency,
            model_dimension,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, lengths):
        features = self.convolutions(features.unsqueeze(1))
        batch, channels, time_steps, frequency = features.shape
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(batch, time_steps, channels * frequency)
        features = self.dropout(self.projection(features))
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return features, lengths


class HindiConformerCTC(nn.Module):
    def __init__(self, output_classes: int):
        super().__init__()
        self.subsampling = ConvSubsampling()
        self.conformer = Conformer(
            input_dim=144,
            num_heads=4,
            ffn_dim=576,
            num_layers=8,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
            use_group_norm=True,
            convolution_first=False,
        )
        self.output_layer = nn.Linear(144, output_classes)

    def forward(self, features, lengths):
        encoded, lengths = self.subsampling(features, lengths)
        encoded, lengths = self.conformer(encoded, lengths)
        return self.output_layer(encoded), lengths


def decode_ctc(token_ids, valid_length: int) -> str:
    characters = []
    previous = None
    for token_id in token_ids[:valid_length]:
        token_id = int(token_id)
        if token_id != BLANK_ID and token_id != previous:
            characters.append(ID_TO_CHAR.get(token_id, ""))
        previous = token_id
    return "".join(characters).strip()


def edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def select_device() -> torch.device:
    requested = os.environ.get("ASR_DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    prepare_files()
    device = select_device()
    print(f"Evaluation device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint.get("phase_epoch", checkpoint.get("epoch", 3)))

    model = HindiConformerCTC(NUMBER_OF_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    dataset = HindiTestDataset(TEST_MANIFEST)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    loss_function = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)

    rows = []
    total_loss = 0.0
    character_errors = 0
    character_count = 0
    word_errors = 0
    word_count = 0
    total_audio_seconds = 0.0
    started = time.time()

    print(f"Checkpoint phase epoch: {checkpoint_epoch}")
    print(f"Test batches: {len(loader)}")
    print("Evaluation started. Progress is printed every 100 batches.")

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            logits, output_lengths = model(features, feature_lengths)

            cpu_logits = logits.float().cpu()
            loss = loss_function(
                cpu_logits.log_softmax(dim=-1).transpose(0, 1),
                batch["targets"],
                output_lengths.cpu(),
                batch["target_lengths"],
            )
            total_loss += float(loss)
            predicted_ids = cpu_logits.argmax(dim=-1)

            for index, reference in enumerate(batch["texts"]):
                prediction = decode_ctc(
                    predicted_ids[index].tolist(),
                    int(output_lengths[index].cpu()),
                )
                ref_words = reference.split()
                hyp_words = prediction.split()
                char_error = edit_distance(reference, prediction)
                word_error = edit_distance(ref_words, hyp_words)
                character_errors += char_error
                character_count += len(reference)
                word_errors += word_error
                word_count += len(ref_words)
                duration = float(batch["durations"][index])
                total_audio_seconds += duration
                rows.append(
                    {
                        "audio_filepath": batch["paths"][index],
                        "duration": duration,
                        "reference": reference,
                        "prediction": prediction,
                        "character_errors": char_error,
                        "word_errors": word_error,
                    }
                )

            if batch_number % 100 == 0 or batch_number == len(loader):
                elapsed_minutes = (time.time() - started) / 60
                running_cer = 100 * character_errors / max(character_count, 1)
                print(
                    f"Batch {batch_number}/{len(loader)} | "
                    f"CER {running_cer:.2f}% | {elapsed_minutes:.1f} min"
                )

    elapsed_seconds = time.time() - started
    test_loss = total_loss / len(loader)
    test_cer = character_errors / character_count
    test_wer = word_errors / word_count
    real_time_factor = elapsed_seconds / total_audio_seconds

    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "checkpoint_phase_epoch": checkpoint_epoch,
        "test_recordings": len(dataset),
        "test_audio_hours": total_audio_seconds / 3600,
        "test_loss": test_loss,
        "test_cer": test_cer,
        "test_wer": test_wer,
        "evaluation_minutes": elapsed_seconds / 60,
        "real_time_factor": real_time_factor,
        "device": str(device),
        "predictions_path": str(PREDICTIONS_PATH),
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\nFINAL 200K TEST RESULTS")
    print(f"Checkpoint phase epoch: {checkpoint_epoch}")
    print(f"Test recordings: {len(dataset)}")
    print(f"Test audio: {total_audio_seconds / 3600:.3f} hours")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test CER: {100 * test_cer:.2f} %")
    print(f"Test WER: {100 * test_wer:.2f} %")
    print(f"Evaluation time: {elapsed_seconds / 60:.2f} minutes")
    print(f"Real-time factor: {real_time_factor:.4f}")
    print("\nExample predictions:")
    for row in rows[:5]:
        print(f"Reference : {row['reference']}")
        print(f"Prediction: {row['prediction']}\n")
    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print(f"Results saved: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
