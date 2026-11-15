import glob
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

from IPython.display import FileLink, display
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.models import Conformer


# -----------------------------------------------------------------------------
# 1. Paths, reproducibility, and GPU checks
# -----------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DATA_ROOT = Path(
    "/kaggle/input/datasets/siddharthroyjj/hindi-asr-100k-private"
)
EPOCH7_ROOT = Path(
    "/kaggle/input/datasets/siddharthroyjj/epoch-7"
)
LOCAL_ROOT = Path("/kaggle/working/hindi_asr_100k")
CHECKPOINT_DIR = Path(
    "/kaggle/working/checkpoints_100k_warmstart"
)

TRAIN_JSON = LOCAL_ROOT / "train_manifest.json"
VALIDATION_JSON = LOCAL_ROOT / "validation_manifest.json"
VOCAB_PATH = DATA_ROOT / "ctc_vocabulary.json"

LAST_CHECKPOINT = CHECKPOINT_DIR / "conformer_100k_last.pt"
BEST_LOSS_CHECKPOINT = CHECKPOINT_DIR / "conformer_100k_best_loss.pt"
BEST_CER_CHECKPOINT = CHECKPOINT_DIR / "conformer_100k_best_cer.pt"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.csv"
BUNDLE_PATH = Path("/kaggle/working/hindi_asr_epoch8_bundle.zip")

if not torch.cuda.is_available():
    raise RuntimeError("Enable GPU T4 x2 in Kaggle Settings first.")

gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    raise RuntimeError(
        f"This cell expects two GPUs, but Kaggle supplied {gpu_count}."
    )

device = torch.device("cuda:0")
print("GPUs available:", gpu_count)
for gpu_index in range(gpu_count):
    print(gpu_index, torch.cuda.get_device_name(gpu_index))


# -----------------------------------------------------------------------------
# 2. Restore the epoch-7 checkpoint into /kaggle/working
# -----------------------------------------------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if not LAST_CHECKPOINT.exists():
    for source_path in EPOCH7_ROOT.rglob("*"):
        if source_path.is_file():
            shutil.copy2(
                source_path,
                CHECKPOINT_DIR / source_path.name,
            )

if not LAST_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Epoch-7 checkpoint was not found at {LAST_CHECKPOINT}."
    )


# -----------------------------------------------------------------------------
# 3. Rebuild manifests using the read-only Kaggle audio paths
# -----------------------------------------------------------------------------

source_train_manifest = DATA_ROOT / "train_manifest.json"
source_validation_manifest = DATA_ROOT / "validation_manifest.json"

for required_path in [
    source_train_manifest,
    source_validation_manifest,
    VOCAB_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Missing required input: {required_path}")

print("Indexing Kaggle audio paths...")
audio_paths = sorted(DATA_ROOT.rglob("*.flac"))

audio_by_name = {}
duplicate_names = set()

for audio_path in audio_paths:
    if audio_path.name in audio_by_name:
        duplicate_names.add(audio_path.name)
    audio_by_name[audio_path.name] = audio_path

if duplicate_names:
    raise ValueError(
        "Duplicate FLAC filenames found: "
        + str(sorted(duplicate_names)[:10])
    )

if len(audio_by_name) != 100567:
    raise ValueError(
        f"Expected 100567 audio files, found {len(audio_by_name)}."
    )

LOCAL_ROOT.mkdir(parents=True, exist_ok=True)


def load_json_lines(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def rebuild_manifest(source_path, destination_path):
    records = load_json_lines(source_path)

    with open(destination_path, "w", encoding="utf-8") as file:
        for record in records:
            old_audio_path = record.get("audio_filepath")
            if old_audio_path is None:
                old_audio_path = record.get("audio_path")

            filename = Path(old_audio_path).name
            actual_audio_path = audio_by_name.get(filename)

            if actual_audio_path is None:
                raise FileNotFoundError(
                    f"Could not locate audio file {filename}."
                )

            record["audio_filepath"] = str(actual_audio_path)

            if "text" not in record and "model_text" in record:
                record["text"] = record["model_text"]

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return records


train_records = rebuild_manifest(
    source_train_manifest,
    TRAIN_JSON,
)
validation_records = rebuild_manifest(
    source_validation_manifest,
    VALIDATION_JSON,
)

if len(train_records) != 100000:
    raise ValueError(
        f"Expected 100000 training records, found {len(train_records)}."
    )

if len(validation_records) != 567:
    raise ValueError(
        f"Expected 567 validation records, found {len(validation_records)}."
    )

print("KAGGLE MANIFESTS READY")
print("Indexed audio files:", len(audio_by_name))
print("Training records:", len(train_records))
print("Validation records:", len(validation_records))


# -----------------------------------------------------------------------------
# 4. Vocabulary, dataset, log-Mel features, and data loaders
# -----------------------------------------------------------------------------

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary_data = json.load(file)

char_to_id = None

for key in [
    "char_to_id",
    "character_to_id",
    "token_to_id",
    "vocabulary",
    "vocab",
]:
    value = vocabulary_data.get(key)
    if isinstance(value, dict):
        candidate = {
            character: int(token_id)
            for character, token_id in value.items()
            if isinstance(character, str)
            and len(character) == 1
        }
        if candidate:
            char_to_id = candidate
            break

if char_to_id is None:
    char_to_id = {
        character: index + 1
        for index, character in enumerate(
            vocabulary_data["characters"]
        )
    }

blank_id = int(vocabulary_data.get("blank_id", 0))
id_to_char = {
    token_id: character
    for character, token_id in char_to_id.items()
}
number_of_classes = max(
    max(char_to_id.values()),
    blank_id,
) + 1


class HindiASRDataset(Dataset):
    def __init__(self, manifest_path):
        self.records = load_json_lines(manifest_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        waveform, sample_rate = sf.read(
            record["audio_filepath"],
            dtype="float32",
        )

        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16000 Hz, received {sample_rate}."
            )

        waveform = torch.from_numpy(waveform)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        text = str(record["text"])
        target = torch.tensor(
            [char_to_id[character] for character in text],
            dtype=torch.long,
        )

        return waveform, target, text


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    power=2.0,
)


def collate_batch(batch):
    waveforms, targets, texts = zip(*batch)

    waveform_lengths = torch.tensor(
        [len(waveform) for waveform in waveforms],
        dtype=torch.long,
    )

    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )

    features = mel_transform(padded_waveforms)
    features = torch.log(
        features.clamp_min(1e-10)
    ).transpose(1, 2)

    feature_lengths = waveform_lengths // 160 + 1

    for index, length in enumerate(feature_lengths):
        valid_length = int(length)
        valid_features = features[index, :valid_length]

        features[index, :valid_length] = (
            valid_features
            - valid_features.mean(dim=0, keepdim=True)
        ) / valid_features.std(
            dim=0,
            keepdim=True,
        ).clamp_min(1e-5)

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "targets": pad_sequence(
            targets,
            batch_first=True,
            padding_value=blank_id,
        ),
        "target_lengths": torch.tensor(
            [len(target) for target in targets],
            dtype=torch.long,
        ),
        "texts": texts,
    }


train_dataset = HindiASRDataset(TRAIN_JSON)
validation_dataset = HindiASRDataset(VALIDATION_JSON)

loader_generator = torch.Generator().manual_seed(
    SEED + 8
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    generator=loader_generator,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_batch,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_batch,
)


# -----------------------------------------------------------------------------
# 5. Conformer-CTC model and the two-GPU-safe attention mask
# -----------------------------------------------------------------------------


def data_parallel_safe_conformer_forward(
    self,
    input,
    lengths,
):
    time_steps = input.size(1)

    encoder_padding_mask = (
        torch.arange(
            time_steps,
            device=input.device,
        ).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )

    x = input.transpose(0, 1)

    for layer in self.conformer_layers:
        x = layer(x, encoder_padding_mask)

    return x.transpose(0, 1), lengths


Conformer.forward = data_parallel_safe_conformer_forward


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        input_features=80,
        model_dimension=144,
    ):
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
            model_dimension * reduced_frequency,
            model_dimension,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, lengths):
        features = self.convolutions(
            features.unsqueeze(1)
        )

        batch, channels, time_steps, frequency = (
            features.shape
        )

        features = features.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        features = features.view(
            batch,
            time_steps,
            channels * frequency,
        )

        features = self.dropout(
            self.projection(features)
        )

        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2

        return features, lengths


class HindiConformerCTC(nn.Module):
    def __init__(self, output_classes):
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
        self.output_layer = nn.Linear(
            144,
            output_classes,
        )

    def forward(self, features, lengths):
        encoded, output_lengths = self.subsampling(
            features,
            lengths,
        )

        encoded, _ = self.conformer(
            encoded,
            output_lengths,
        )

        return self.output_layer(encoded), output_lengths


checkpoint = torch.load(
    LAST_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

continuation_epoch = int(
    checkpoint.get(
        "continuation_epoch",
        checkpoint.get("epoch", -1),
    )
)

if continuation_epoch != 7:
    raise ValueError(
        "Expected the epoch-7 checkpoint, but found epoch "
        f"{continuation_epoch}."
    )

saved_state = checkpoint["model_state_dict"]

if any(key.startswith("module.") for key in saved_state):
    saved_state = {
        key.removeprefix("module."): value
        for key, value in saved_state.items()
    }

model = HindiConformerCTC(
    number_of_classes
)
model.load_state_dict(saved_state, strict=True)
model.to(device)

parallel_model = nn.DataParallel(
    model,
    device_ids=[0, 1],
)

ctc_loss_function = nn.CTCLoss(
    blank=blank_id,
    reduction="mean",
    zero_infinity=True,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


# -----------------------------------------------------------------------------
# 6. Optimizer, the original 12-epoch schedule, and decoding metrics
# -----------------------------------------------------------------------------

TOTAL_CONTINUATION_EPOCHS = 12
TARGET_CONTINUATION_EPOCH = 8
BASE_LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4

updates_per_epoch = math.ceil(
    len(train_loader)
    / GRADIENT_ACCUMULATION_STEPS
)
warmup_steps = updates_per_epoch
total_update_steps = (
    TOTAL_CONTINUATION_EPOCHS
    * updates_per_epoch
)


def learning_rate_multiplier(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (
        (step - warmup_steps)
        / max(1, total_update_steps - warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return 0.1 + 0.9 * cosine


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LEARNING_RATE,
    weight_decay=1e-3,
)
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=learning_rate_multiplier,
)
scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)

# LambdaLR initialization can modify the optimizer LR. Restore the exact saved LR.
for parameter_group, saved_group in zip(
    optimizer.param_groups,
    checkpoint["optimizer_state_dict"]["param_groups"],
):
    parameter_group["lr"] = saved_group["lr"]

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=True,
)

if checkpoint.get("scaler_state_dict"):
    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

global_step = int(checkpoint.get("global_step", 0))
best_validation_loss = float(
    checkpoint.get("best_validation_loss", float("inf"))
)
best_validation_cer = float(
    checkpoint.get("best_validation_cer", float("inf"))
)
history = list(checkpoint.get("history", []))


def apply_spec_augment(features, lengths):
    augmented = features.clone()
    batch_size, _, feature_count = augmented.shape

    for index in range(batch_size):
        valid_length = int(lengths[index])

        for _ in range(2):
            width = random.randint(0, 8)
            if width > 0:
                start = random.randint(
                    0,
                    feature_count - width,
                )
                augmented[
                    index,
                    :valid_length,
                    start:start + width,
                ] = 0

        maximum_time_width = min(
            40,
            max(1, int(valid_length * 0.05)),
        )

        for _ in range(2):
            width = random.randint(
                0,
                maximum_time_width,
            )

            if width > 0 and valid_length > width:
                start = random.randint(
                    0,
                    valid_length - width,
                )
                augmented[
                    index,
                    start:start + width,
                    :,
                ] = 0

    return augmented


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current = [reference_index]

        for prediction_index, prediction_item in enumerate(
            prediction,
            start=1,
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

    for sequence, length in zip(
        sequences,
        output_lengths,
    ):
        characters = []
        previous_token = None

        for token in sequence[:int(length)].tolist():
            if (
                token != blank_id
                and token != previous_token
                and token in id_to_char
            ):
                characters.append(id_to_char[token])

            previous_token = token

        predictions.append(
            "".join(characters).strip()
        )

    return predictions


def calculate_error_rates(references, predictions):
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0

    for reference, prediction in zip(
        references,
        predictions,
    ):
        character_errors += edit_distance(
            list(reference),
            list(prediction),
        )
        reference_characters += len(reference)

        reference_word_list = reference.split()
        prediction_word_list = prediction.split()

        word_errors += edit_distance(
            reference_word_list,
            prediction_word_list,
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
            features = batch["features"].to(
                device,
                non_blocking=True,
            )
            feature_lengths = batch[
                "feature_lengths"
            ].to(device)
            targets = batch["targets"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                logits, output_lengths = parallel_model(
                    features,
                    feature_lengths,
                )

            loss = ctc_loss_function(
                logits.float()
                .log_softmax(dim=-1)
                .transpose(0, 1),
                targets,
                output_lengths.cpu(),
                batch["target_lengths"],
            )

            total_loss += float(loss.cpu())

            predictions.extend(
                greedy_decode(
                    logits.cpu(),
                    output_lengths.cpu(),
                )
            )
            references.extend(batch["texts"])

    cer, wer = calculate_error_rates(
        references,
        predictions,
    )

    return {
        "loss": total_loss / len(validation_loader),
        "cer": cer,
        "wer": wer,
        "reference": references[0],
        "prediction": predictions[0],
    }


print("\n100K EPOCH-8 RESUME SETUP COMPLETE")
print("Training recordings:", len(train_dataset))
print("Validation recordings:", len(validation_dataset))
print("Training batches:", len(train_loader))
print("Effective batch size:", 8 * GRADIENT_ACCUMULATION_STEPS)
print("Trainable parameters:", round(parameter_count / 1_000_000, 2), "million")
print("Resuming after epoch:", continuation_epoch)
print("Starting learning rate:", optimizer.param_groups[0]["lr"])
print("GPUs used:", len(parallel_model.device_ids))


# -----------------------------------------------------------------------------
# 7. Train exactly continuation epoch 8
# -----------------------------------------------------------------------------

epoch = TARGET_CONTINUATION_EPOCH
epoch_start = time.time()

parallel_model.train()
optimizer.zero_grad(set_to_none=True)
total_training_loss = 0.0

for batch_number, batch in enumerate(
    train_loader,
    start=1,
):
    features = batch["features"].to(
        device,
        non_blocking=True,
    )
    feature_lengths = batch[
        "feature_lengths"
    ].to(device)
    targets = batch["targets"].to(
        device,
        non_blocking=True,
    )

    features = apply_spec_augment(
        features,
        feature_lengths,
    )

    with torch.amp.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        logits, output_lengths = parallel_model(
            features,
            feature_lengths,
        )

        loss = ctc_loss_function(
            logits.float()
            .log_softmax(dim=-1)
            .transpose(0, 1),
            targets,
            output_lengths.cpu(),
            batch["target_lengths"],
        )

        scaled_loss = (
            loss / GRADIENT_ACCUMULATION_STEPS
        )

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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

    total_training_loss += float(
        loss.detach().cpu()
    )

    if batch_number % 1250 == 0:
        print(
            f"Epoch {epoch} | "
            f"batch {batch_number}/{len(train_loader)} | "
            f"loss {float(loss.detach().cpu()):.3f}"
        )

training_loss = (
    total_training_loss / len(train_loader)
)
validation_result = evaluate_validation()
epoch_minutes = (time.time() - epoch_start) / 60
learning_rate = scheduler.get_last_lr()[0]

history.append({
    "continuation_epoch": epoch,
    "training_loss": training_loss,
    "validation_loss": validation_result["loss"],
    "validation_cer": validation_result["cer"],
    "validation_wer": validation_result["wer"],
    "learning_rate": learning_rate,
    "minutes": epoch_minutes,
})

loss_improved = (
    validation_result["loss"]
    < best_validation_loss
)
cer_improved = (
    validation_result["cer"]
    < best_validation_cer
)

if loss_improved:
    best_validation_loss = validation_result["loss"]

if cer_improved:
    best_validation_cer = validation_result["cer"]

checkpoint_to_save = {
    "source_checkpoint_epoch": int(
        checkpoint.get("source_checkpoint_epoch", 30)
    ),
    "continuation_epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "best_validation_loss": best_validation_loss,
    "best_validation_cer": best_validation_cer,
    "history": history,
}

torch.save(
    checkpoint_to_save,
    LAST_CHECKPOINT,
)

if loss_improved:
    torch.save(
        checkpoint_to_save,
        BEST_LOSS_CHECKPOINT,
    )

if cer_improved:
    torch.save(
        checkpoint_to_save,
        BEST_CER_CHECKPOINT,
    )

pd.DataFrame(history).to_csv(
    HISTORY_PATH,
    index=False,
)

print(f"\nContinuation epoch {epoch}/12")
print("Training loss:", round(training_loss, 4))
print(
    "Validation loss:",
    round(validation_result["loss"], 4),
)
print(
    "Validation CER:",
    round(validation_result["cer"] * 100, 2),
    "%",
)
print(
    "Validation WER:",
    round(validation_result["wer"] * 100, 2),
    "%",
)
print(
    "Learning rate:",
    f"{learning_rate:.7f}",
)
print(
    "Epoch time:",
    round(epoch_minutes, 2),
    "minutes",
)
print("Best-loss updated:", loss_improved)
print("Best-CER updated:", cer_improved)
print("Reference:", validation_result["reference"])
print("Prediction:", validation_result["prediction"])


# -----------------------------------------------------------------------------
# 8. Create the downloadable epoch-8 backup
# -----------------------------------------------------------------------------

with zipfile.ZipFile(
    BUNDLE_PATH,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for path in [
        LAST_CHECKPOINT,
        BEST_LOSS_CHECKPOINT,
        BEST_CER_CHECKPOINT,
        HISTORY_PATH,
    ]:
        if path.exists():
            archive.write(
                path,
                arcname=path.name,
            )

print("\nKAGGLE EPOCH-8 STAGE FINISHED")
print("Latest checkpoint:", LAST_CHECKPOINT)
print("Backup bundle:", BUNDLE_PATH)
print(
    "Backup size:",
    round(BUNDLE_PATH.stat().st_size / 1024**2, 2),
    "MB",
)

display(FileLink(str(BUNDLE_PATH)))
