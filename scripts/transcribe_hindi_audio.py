#!/usr/bin/env python3
"""Transcribe one Hindi audio file with the final 200K Conformer-CTC model."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from evaluate_hindi_asr_200k import (
    BLANK_ID,
    CHECKPOINT_PATH,
    HindiConformerCTC,
    ID_TO_CHAR,
    MEL_TRANSFORM,
    NUMBER_OF_CLASSES,
    prepare_files,
    select_device,
)


def read_audio(path: Path):
    """Read common audio formats, using macOS conversion when needed."""
    try:
        waveform, sample_rate = sf.read(path, dtype="float32")
        return torch.from_numpy(waveform), sample_rate
    except Exception:
        temporary = Path(tempfile.gettempdir()) / "hindi_asr_converted.wav"
        command = [
            "/usr/bin/afconvert",
            str(path),
            str(temporary),
            "-f", "WAVE",
            "-d", "LEI16@16000",
            "-c", "1",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Could not read or convert the audio file. "
                "Use a WAV, FLAC, or M4A file.\n" + result.stderr
            )
        waveform, sample_rate = sf.read(temporary, dtype="float32")
        temporary.unlink(missing_ok=True)
        return torch.from_numpy(waveform), sample_rate


def decode(token_ids, valid_length):
    characters = []
    previous = None
    for token_id in token_ids[:valid_length]:
        token_id = int(token_id)
        if token_id != BLANK_ID and token_id != previous:
            characters.append(ID_TO_CHAR.get(token_id, ""))
        previous = token_id
    return "".join(characters).strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe one Hindi audio file")
    parser.add_argument("audio", type=Path, help="Path to a WAV, FLAC, M4A, MP3, or Voice Memo file")
    arguments = parser.parse_args()
    audio_path = arguments.audio.expanduser().resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    prepare_files()
    device = select_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = HindiConformerCTC(NUMBER_OF_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    waveform, sample_rate = read_audio(audio_path)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    duration = waveform.numel() / sample_rate
    if duration < 0.5:
        raise ValueError("The recording is shorter than 0.5 seconds.")
    if duration > 30:
        print("Warning: the model was trained mainly on clips of 30 seconds or less.")

    features = MEL_TRANSFORM(waveform.unsqueeze(0))
    features = torch.log(features.clamp_min(1e-10)).transpose(1, 2)
    feature_length = torch.tensor([features.size(1)], dtype=torch.long)
    valid = features[0, : int(feature_length[0])]
    features[0, : int(feature_length[0])] = (
        valid - valid.mean(dim=0, keepdim=True)
    ) / valid.std(dim=0, keepdim=True).clamp_min(1e-5)

    with torch.inference_mode():
        logits, output_lengths = model(features.to(device), feature_length.to(device))
        predicted_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    transcript = decode(predicted_ids, int(output_lengths[0].cpu()))
    print("\nHINDI TRANSCRIPT")
    print(transcript if transcript else "[No speech was recognized]")
    print(f"\nAudio duration: {duration:.2f} seconds")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
