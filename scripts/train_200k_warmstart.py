"""Warm-start the 200K Hindi Conformer-CTC stage on Kaggle.

The script reproduces the project's final training phase:

* 200,000 speaker-filtered recordings (381.88 hours)
* fixed 567-record speaker-disjoint development set
* 100K epoch-12 checkpoint used only to initialize model weights
* four planned 200K phase epochs, peak learning rate 3e-5
* batch size 8 with four-step gradient accumulation (effective batch 32)
* safe checkpoint and ZIP backup after every completed epoch

By default one phase epoch is run per invocation. This is intentional: one
200K epoch took roughly two hours on Kaggle T4 x2, and saving after every epoch
protects the run from notebook timeouts. Attach the previous epoch bundle as a
Kaggle input before starting the next invocation.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.models import Conformer

from prepare_kaggle_200k_manifests import (
    BASE_100K_ROOT,
    TRAIN_JSON,
    VALIDATION_JSON,
    load_json_lines,
    prepare_manifests,
)


# -----------------------------------------------------------------------------
# 1. Configuration and reproducibility
# -----------------------------------------------------------------------------

SEED = 42
TOTAL_PHASE_EPOCHS = int(os.environ.get("HINDI_ASR_200K_EPOCHS", "4"))
EPOCHS_THIS_RUN = int(os.environ.get("HINDI_ASR_EPOCHS_THIS_RUN", "1"))
BATCH_SIZE = int(os.environ.get("HINDI_ASR_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("HINDI_ASR_GRAD_ACCUM", "4")
)
PEAK_LEARNING_RATE = float(
    os.environ.get("HINDI_ASR_200K_PEAK_LR", "3e-5")
)

SOURCE_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "HINDI_ASR_100K_CHECKPOINT_ROOT",
        "/kaggle/input/datasets/siddharthroyjj/epoch-12",
    )
)
CHECKPOINT_DIR = Path(
    os.environ.get(
        "HINDI_ASR_200K_CHECKPOINT_DIR",
        "/kaggle/working/checkpoints_200k_warmstart",
    )
)

LAST_CHECKPOINT = CHECKPOINT_DIR / "conformer_200k_last.pt"
BEST_LOSS_CHECKPOINT = (
    CHECKPOINT_DIR / "conformer_200k_best_loss.pt"
)
BEST_CER_CHECKPOINT = (
    CHECKPOINT_DIR / "conformer_200k_best_cer.pt"
)
HISTORY_PATH = CHECKPOINT_DIR / "training_history.csv"
VOCAB_PATH = BASE_100K_ROOT / "ctc_vocabulary.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before training.")

device = torch.device("cuda:0")
gpu_count = torch.cuda.device_count()
print("GPUs available:", gpu_count)
for gpu_index in range(gpu_count):
    print(gpu_index, torch.cuda.get_device_name(gpu_index))

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
prepare_manifests()


# -----------------------------------------------------------------------------
# 2. Vocabulary, datasets, and log-Mel feature extraction
# -----------------------------------------------------------------------------

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing vocabulary: {VOCAB_PATH}")

with VOCAB_PATH.open("r", encoding="utf-8") as file:
    vocabulary_data = json.load(file)

char_to_id = None
for key in (
    "char_to_id",
    "character_to_id",
    "token_to_id",
    "vocabulary",
    "vocab",
):
    value = vocabulary_data.get(key)
    if isinstance(value, dict):
        candidate = {
            character: int(token_id)
            for character, token_id in value.items()
            if isinstance(character, str) and len(character) == 1
        }
        if candidate:
            char_to_id = candidate
            break

if char_to_id is None:
    char_to_id = {
        character: index + 1
        for index, character in enumerate(vocabulary_data["characters"])
    }

blank_id = int(vocabulary_data.get("blank_id", 0))
id_to_char = {
    token_id: character for character, token_id in char_to_id.items()
}
number_of_classes = max(max(char_to_id.values()), blank_id) + 1


class HindiASRDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.records = load_json_lines(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        waveform, sample_rate = sf.read(
            record["audio_filepath"], dtype="float32"
        )
        if sample_rate != 16_000:
            raise ValueError(
                f"Expected 16000 Hz, received {sample_rate}."
            )

        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        text = str(record["text"])
        try:
            target = torch.tensor(
                [char_to_id[character] for character in text],
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(
                f"Transcript contains unknown character {error.args[0]!r}."
            ) from error

        return waveform, target, text


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16_000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    power=2.0,
)


def collate_batch(batch):
    waveforms, targets, texts = zip(*batch)
    waveform_lengths = torch.tensor(
        [len(waveform) for waveform in waveforms], dtype=torch.long
    )
    padded_waveforms = pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )

    features = torch.log(
        mel_transform(padded_waveforms).clamp_min(1e-10)
    ).transpose(1, 2)
    feature_lengths = waveform_lengths // 160 + 1

    for index, length in enumerate(feature_lengths):
        valid_length = int(length)
        valid_features = features[index, :valid_length]
        features[index, :valid_length] = (
            valid_features - valid_features.mean(dim=0, keepdim=True)
        ) / valid_features.std(dim=0, keepdim=True).clamp_min(1e-5)

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "targets": pad_sequence(
            targets, batch_first=True, padding_value=blank_id
        ),
        "target_lengths": torch.tensor(
            [len(target) for target in targets], dtype=torch.long
        ),
        "texts": texts,
    }


train_dataset = HindiASRDataset(TRAIN_JSON)
validation_dataset = HindiASRDataset(VALIDATION_JSON)

if len(train_dataset) != 200_000:
    raise ValueError(
        f"Expected 200000 training recordings, found {len(train_dataset)}."
    )
if len(validation_dataset) != 567:
    raise ValueError(
        "Expected 567 validation recordings, found "
        f"{len(validation_dataset)}."
    )


def build_train_loader(seed_offset: int) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED + seed_offset)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_batch,
    )


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_batch,
)


# -----------------------------------------------------------------------------
# 3. Conformer-CTC architecture
# -----------------------------------------------------------------------------


def data_parallel_safe_conformer_forward(self, input, lengths):
    time_steps = input.size(1)
    encoder_padding_mask = (
        torch.arange(time_steps, device=input.device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    x = input.transpose(0, 1)
    for layer in self.conformer_layers:
        x = layer(x, encoder_padding_mask)
    return x.transpose(0, 1), lengths


Conformer.forward = data_parallel_safe_conformer_forward


class ConvSubsampling(nn.Module):
    def __init__(self, input_features=80, model_dimension=144):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                1,
                model_dimension,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
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
            model_dimension * reduced_frequency, model_dimension
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, lengths):
        features = self.convolutions(features.unsqueeze(1))
        batch, channels, time_steps, frequency = features.shape
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(
            batch, time_steps, channels * frequency
        )
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
        encoded, output_lengths = self.subsampling(features, lengths)
        encoded, _ = self.conformer(encoded, output_lengths)
        return self.output_layer(encoded), output_lengths


# -----------------------------------------------------------------------------
# 4. Checkpoint selection and resume handling
# -----------------------------------------------------------------------------


def checkpoint_epoch(path: Path, key: str) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get(key, -1))


def find_100k_source_checkpoint() -> Path:
    preferred_names = (
        "conformer_100k_best_cer.pt",
        "conformer_100k_last.pt",
        "conformer_100k_best_cer_epoch12.pt",
    )
    for name in preferred_names:
        matches = list(SOURCE_CHECKPOINT_ROOT.rglob(name))
        for path in matches:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )
            epoch = int(
                checkpoint.get(
                    "continuation_epoch", checkpoint.get("epoch", -1)
                )
            )
            if epoch == 12:
                return path
    raise FileNotFoundError(
        "Could not find a 100K continuation-epoch-12 checkpoint under "
        f"{SOURCE_CHECKPOINT_ROOT}."
    )


def find_200k_resume_checkpoint() -> Path | None:
    explicit = os.environ.get("HINDI_ASR_200K_RESUME")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    candidates = []
    if LAST_CHECKPOINT.exists():
        candidates.append(LAST_CHECKPOINT)
    candidates.extend(
        Path("/kaggle/input").rglob("conformer_200k_last.pt")
    )
    if not candidates:
        return None

    ranked = []
    for path in candidates:
        try:
            ranked.append((checkpoint_epoch(path, "phase_epoch"), path))
        except Exception as error:
            print(f"Skipping unreadable checkpoint {path}: {error}")
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


resume_path = find_200k_resume_checkpoint()
source_path = None if resume_path else find_100k_source_checkpoint()
load_path = resume_path or source_path
checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

if resume_path:
    for filename, destination in (
        ("conformer_200k_best_loss.pt", BEST_LOSS_CHECKPOINT),
        ("conformer_200k_best_cer.pt", BEST_CER_CHECKPOINT),
        ("training_history.csv", HISTORY_PATH),
    ):
        source = resume_path.parent / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

saved_state = checkpoint["model_state_dict"]
if any(key.startswith("module.") for key in saved_state):
    saved_state = {
        key.removeprefix("module."): value
        for key, value in saved_state.items()
    }

model = HindiConformerCTC(number_of_classes)
model.load_state_dict(saved_state, strict=True)
model.to(device)

if gpu_count > 1:
    parallel_model = nn.DataParallel(
        model, device_ids=list(range(gpu_count))
    )
else:
    parallel_model = model

ctc_loss_function = nn.CTCLoss(
    blank=blank_id, reduction="mean", zero_infinity=True
)


# -----------------------------------------------------------------------------
# 5. Optimizer, phase schedule, augmentation, and metrics
# -----------------------------------------------------------------------------

reference_loader = build_train_loader(seed_offset=200)
updates_per_epoch = math.ceil(
    len(reference_loader) / GRADIENT_ACCUMULATION_STEPS
)
total_update_steps = TOTAL_PHASE_EPOCHS * updates_per_epoch


def learning_rate_multiplier(step: int) -> float:
    progress = min(max(step / max(1, total_update_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.1 + 0.9 * cosine


optimizer = torch.optim.AdamW(
    model.parameters(), lr=PEAK_LEARNING_RATE, weight_decay=1e-3
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=learning_rate_multiplier
)
scaler = torch.amp.GradScaler("cuda", enabled=True)

if resume_path:
    completed_phase_epoch = int(checkpoint["phase_epoch"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    global_step = int(checkpoint.get("global_step", 0))
    phase_history = list(checkpoint.get("phase_history", []))
    best_validation_loss = float(
        checkpoint.get("best_validation_loss", float("inf"))
    )
    best_validation_cer = float(
        checkpoint.get("best_validation_cer", float("inf"))
    )
    source_checkpoint_epoch = int(
        checkpoint.get("source_checkpoint_epoch", 12)
    )
else:
    source_epoch = int(
        checkpoint.get(
            "continuation_epoch", checkpoint.get("epoch", -1)
        )
    )
    if source_epoch != 12:
        raise ValueError(
            f"Expected 100K epoch 12, found epoch {source_epoch}."
        )
    completed_phase_epoch = 0
    global_step = 0
    phase_history = []
    best_validation_loss = float("inf")
    best_validation_cer = float("inf")
    source_checkpoint_epoch = 12


def apply_spec_augment(features, lengths):
    augmented = features.clone()
    batch_size, _, feature_count = augmented.shape
    for index in range(batch_size):
        valid_length = int(lengths[index])
        for _ in range(2):
            width = random.randint(0, 8)
            if width:
                start = random.randint(0, feature_count - width)
                augmented[index, :valid_length, start : start + width] = 0

        maximum_time_width = min(
            40, max(1, int(valid_length * 0.05))
        )
        for _ in range(2):
            width = random.randint(0, maximum_time_width)
            if width and valid_length > width:
                start = random.randint(0, valid_length - width)
                augmented[index, start : start + width, :] = 0
    return augmented


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_item in enumerate(
            prediction, start=1
        ):
            current.append(
                min(
                    current[prediction_index - 1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def greedy_decode(logits, output_lengths):
    sequences = logits.argmax(dim=-1)
    predictions = []
    for sequence, length in zip(sequences, output_lengths):
        characters = []
        previous_token = None
        for token in sequence[: int(length)].tolist():
            if (
                token != blank_id
                and token != previous_token
                and token in id_to_char
            ):
                characters.append(id_to_char[token])
            previous_token = token
        predictions.append("".join(characters).strip())
    return predictions


def calculate_error_rates(references, predictions):
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0
    for reference, prediction in zip(references, predictions):
        character_errors += edit_distance(list(reference), list(prediction))
        reference_characters += len(reference)
        reference_word_list = reference.split()
        prediction_word_list = prediction.split()
        word_errors += edit_distance(
            reference_word_list, prediction_word_list
        )
        reference_words += len(reference_word_list)
    return (
        character_errors / max(1, reference_characters),
        word_errors / max(1, reference_words),
    )


def evaluate_validation():
    parallel_model.eval()
    total_loss = 0.0
    references = []
    predictions = []
    with torch.inference_mode():
        for batch in validation_loader:
            features = batch["features"].to(device, non_blocking=True)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, output_lengths = parallel_model(
                    features, feature_lengths
                )
            loss = ctc_loss_function(
                logits.float().log_softmax(dim=-1).transpose(0, 1),
                targets,
                output_lengths.cpu(),
                batch["target_lengths"],
            )
            total_loss += float(loss.cpu())
            predictions.extend(
                greedy_decode(logits.cpu(), output_lengths.cpu())
            )
            references.extend(batch["texts"])

    cer, wer = calculate_error_rates(references, predictions)
    return {
        "loss": total_loss / len(validation_loader),
        "cer": cer,
        "wer": wer,
        "reference": references[0],
        "prediction": predictions[0],
    }


if not resume_path:
    baseline = evaluate_validation()
    best_validation_loss = baseline["loss"]
    best_validation_cer = baseline["cer"]
    print("\nBASELINE FROM 100K EPOCH 12")
    print("Validation loss:", round(baseline["loss"], 4))
    print("Validation CER:", round(baseline["cer"] * 100, 2), "%")
    print("Validation WER:", round(baseline["wer"] * 100, 2), "%")

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n200K WARM-START SETUP COMPLETE")
print("Loaded checkpoint:", load_path)
print("Completed 200K phase epoch:", completed_phase_epoch)
print("Training recordings:", len(train_dataset))
print("Validation recordings:", len(validation_dataset))
print("Training batches:", len(reference_loader))
print("Updates per epoch:", updates_per_epoch)
print(
    "Effective batch size:",
    BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
)
print("Trainable parameters:", round(parameter_count / 1_000_000, 2), "million")
print("GPUs used:", gpu_count)


# -----------------------------------------------------------------------------
# 6. Train and save every completed 200K phase epoch
# -----------------------------------------------------------------------------

first_epoch = completed_phase_epoch + 1
last_epoch = min(
    TOTAL_PHASE_EPOCHS,
    completed_phase_epoch + EPOCHS_THIS_RUN,
)

if first_epoch > TOTAL_PHASE_EPOCHS:
    print("All planned 200K phase epochs are already complete.")
    raise SystemExit(0)

for phase_epoch in range(first_epoch, last_epoch + 1):
    train_loader = build_train_loader(seed_offset=200 + phase_epoch)
    epoch_start = time.time()
    parallel_model.train()
    optimizer.zero_grad(set_to_none=True)
    total_training_loss = 0.0

    for batch_number, batch in enumerate(train_loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device)
        targets = batch["targets"].to(device, non_blocking=True)
        features = apply_spec_augment(features, feature_lengths)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, output_lengths = parallel_model(
                features, feature_lengths
            )
            loss = ctc_loss_function(
                logits.float().log_softmax(dim=-1).transpose(0, 1),
                targets,
                output_lengths.cpu(),
                batch["target_lengths"],
            )
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss at batch {batch_number}."
            )

        scaler.scale(scaled_loss).backward()
        should_update = (
            batch_number % GRADIENT_ACCUMULATION_STEPS == 0
            or batch_number == len(train_loader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        total_training_loss += float(loss.detach().cpu())
        if batch_number % 2_500 == 0:
            print(
                f"200K epoch {phase_epoch} | "
                f"batch {batch_number}/{len(train_loader)} | "
                f"loss {float(loss.detach().cpu()):.3f}"
            )

    training_loss = total_training_loss / len(train_loader)
    validation_result = evaluate_validation()
    epoch_minutes = (time.time() - epoch_start) / 60
    learning_rate = optimizer.param_groups[0]["lr"]

    loss_improved = validation_result["loss"] < best_validation_loss
    cer_improved = validation_result["cer"] < best_validation_cer
    if loss_improved:
        best_validation_loss = validation_result["loss"]
    if cer_improved:
        best_validation_cer = validation_result["cer"]

    phase_history.append(
        {
            "phase_epoch": phase_epoch,
            "training_loss": training_loss,
            "validation_loss": validation_result["loss"],
            "validation_cer": validation_result["cer"],
            "validation_wer": validation_result["wer"],
            "learning_rate": learning_rate,
            "minutes": epoch_minutes,
        }
    )

    checkpoint_to_save = {
        "source_checkpoint_epoch": source_checkpoint_epoch,
        "phase_epoch": phase_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_validation_loss": best_validation_loss,
        "best_validation_cer": best_validation_cer,
        "phase_history": phase_history,
        "training_recordings": len(train_dataset),
        "training_audio_hours": 381.88,
    }

    torch.save(checkpoint_to_save, LAST_CHECKPOINT)
    if loss_improved:
        torch.save(checkpoint_to_save, BEST_LOSS_CHECKPOINT)
    if cer_improved:
        torch.save(checkpoint_to_save, BEST_CER_CHECKPOINT)
    pd.DataFrame(phase_history).to_csv(HISTORY_PATH, index=False)

    bundle_path = Path(
        f"/kaggle/working/hindi_asr_200k_epoch{phase_epoch}_bundle.zip"
    )
    with zipfile.ZipFile(
        bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in (
            LAST_CHECKPOINT,
            BEST_LOSS_CHECKPOINT,
            BEST_CER_CHECKPOINT,
            HISTORY_PATH,
        ):
            if path.exists():
                archive.write(path, arcname=path.name)

    print(f"\n200K continuation epoch {phase_epoch}/{TOTAL_PHASE_EPOCHS}")
    print("Training loss:", round(training_loss, 4))
    print("Validation loss:", round(validation_result["loss"], 4))
    print("Validation CER:", round(validation_result["cer"] * 100, 2), "%")
    print("Validation WER:", round(validation_result["wer"] * 100, 2), "%")
    print("Learning rate:", f"{learning_rate:.8f}")
    print("Epoch time:", round(epoch_minutes, 2), "minutes")
    print("Best-loss updated:", loss_improved)
    print("Best-CER updated:", cer_improved)
    print("Reference:", validation_result["reference"])
    print("Prediction:", validation_result["prediction"])
    print("Latest checkpoint:", LAST_CHECKPOINT)
    print("Backup bundle:", bundle_path)
    print(
        "Backup size:",
        round(bundle_path.stat().st_size / 1024**2, 2),
        "MB",
    )

print("\n200K TRAINING INVOCATION FINISHED SAFELY")
