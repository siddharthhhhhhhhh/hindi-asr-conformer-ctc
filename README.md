# Hindi Automatic Speech Recognition with Conformer–CTC

This repository contains the code, evaluation results, and documentation for a
Hindi automatic speech recognition system trained on the Hindi portion of
AI4Bharat IndicVoices. The system converts 16 kHz mono speech into Devanagari
text; it does not perform translation or question answering.

## Final result

The final model was continued from the smaller controlled experiments and
trained on 200,000 speaker-filtered recordings (381.88 hours). Evaluation used
a fixed speaker-disjoint test set containing 5,897 recordings (9.495 hours)
from 132 speakers absent from training.

| Model stage | Training clips | Audio hours | Test CER | Test WER |
|---|---:|---:|---:|---:|
| 10-hour | 5,722 | 9.494 | 46.67% | 91.37% |
| 25-hour | 14,352 | 23.648 | 36.33% | 78.35% |
| 50-hour | 28,699 | 47.465 | 28.45% | 64.19% |
| Final 200K | 200,000 | 381.88 | **22.80%** | **53.71%** |

Lower CER and WER are better. All reported test results use the same unseen-
speaker evaluation set.

## Architecture

```text
16 kHz waveform
  -> 80-bin log-Mel features (25 ms window, 10 ms hop)
  -> two 3x3 stride-2 Conv2D layers
  -> linear projection to dimension 144
  -> eight Conformer blocks
       4 attention heads
       feed-forward dimension 576
       depthwise convolution kernel 31
  -> linear layer with 67 outputs
       66 character tokens + CTC blank
  -> greedy CTC decoding
  -> Devanagari transcript
```

The model has approximately 4.5 million trainable parameters and does not use
wav2vec, Whisper, or an external language model.

## Repository structure

```text
notebooks/
  01_base_training_10h_25h_50h.ipynb
scripts/
  kaggle_continue_training_100k.py
  prepare_kaggle_200k_manifests.py
  train_200k_warmstart.py
  evaluate_hindi_asr_200k.py
  transcribe_hindi_audio.py
results/
  final_results.json
  validation_progress.csv
  test_examples.csv
  figures/
docs/
  Hindi_ASR_Project_Report.docx
  Hindi_ASR_Project_Report.pdf
  Hindi_ASR_Project_Presentation.pptx
```

The notebook documents the controlled 10/25/50-hour experiments. The larger
100K and 200K stages were warm-start continuations from saved checkpoints. The
final evaluation and single-file transcription scripts use the 200K checkpoint.

## Train the 200K stage on Kaggle

Attach these three private Kaggle datasets to a GPU notebook:

```text
siddharthroyjj/hindi-asr-100k-private
siddharthroyjj/hindi-asr-extra-100k-private
siddharthroyjj/epoch-12
```

Then run:

```bash
python scripts/prepare_kaggle_200k_manifests.py
python scripts/train_200k_warmstart.py
```

The preparation script indexes both read-only audio inputs, reconstructs the
200,000-record training manifest, and rechecks duplicate rows and speaker
overlap. The training script loads the 100K epoch-12 weights, uses the same
4.5-million-parameter Conformer-CTC architecture, and saves a checkpoint plus a
downloadable ZIP after every phase epoch. It defaults to one roughly two-hour
epoch per invocation to reduce the risk of losing work to a notebook timeout.
Attach the previous epoch bundle before the next run; the latest
`conformer_200k_last.pt` is detected automatically.

The planned configuration is four phase epochs, batch size 8, four-step
gradient accumulation (effective batch 32), AdamW, mixed precision, gradient
clipping, SpecAugment, and a `3e-5` peak learning rate with cosine decay. The
submitted final checkpoint was the best model available after three completed
phase epochs.

## Data preparation

1. Request access to `ai4bharat/IndicVoices` on Hugging Face.
2. Select the Hindi configuration.
3. Use normalized transcripts and speaker identifiers.
4. Rebuild train, validation, and test partitions by speaker identity.
5. Remove clips shorter than 0.5 seconds, empty/unintelligible transcripts,
   unexpected characters, and CTC-incompatible examples.
6. Retain clips up to 30 seconds.
7. Build a 66-character vocabulary and reserve ID 0 for the CTC blank.

The raw IndicVoices audio is not redistributed in this repository because the
dataset is gated and subject to its own access conditions.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required external artifacts

The final checkpoint, vocabulary, and test-audio bundle are intentionally not
committed to ordinary Git history. The checkpoint ZIP is about 141 MB, which is
larger than GitHub's normal 100 MB per-file limit. Publish it using Git LFS or a
GitHub Release, then record the download URL and SHA-256 value in
`MODEL_ARTIFACTS.md`.

Expected filenames used by the supplied evaluation script are:

```text
hindi_asr_200k_epoch3_bundle.zip
hindi_asr_test_10h_parts-20261109T163236Z-1-001.zip
ctc_vocabulary.json
```

## Evaluate the final model

Place the three files above in `~/Downloads`, then run:

```bash
python scripts/evaluate_hindi_asr_200k.py
```

The script restores the test package and checkpoint, computes CTC loss, CER and
WER, and writes per-recording predictions and a JSON result file under
`~/Hindi_ASR_Final_Evaluation`.

## Transcribe a new recording

After placing the checkpoint and vocabulary in the locations described above:

```bash
python scripts/transcribe_hindi_audio.py path/to/hindi_audio.wav
```

The script accepts common audio formats, converts stereo to mono, resamples to
16 kHz when necessary, computes the same log-Mel features, and applies greedy
CTC decoding. Recordings between 0.5 and 30 seconds are closest to the training
conditions.

## Reproducing the reported experiment

The full experiment requires the gated dataset, prepared speaker-disjoint
manifests, and substantial GPU time. The reproducible sequence is:

1. Run the base notebook for the nested 10/25/50-hour experiments.
2. Save model, optimizer, scheduler, scaler, epoch, and history in every
   checkpoint.
3. Warm-start the 100K stage from the selected 50-hour checkpoint.
4. Warm-start the 200K stage from the 100K epoch-12 checkpoint.
5. Keep the 567-record development set and 5,897-record test set unchanged.
6. Select checkpoints using development metrics only.
7. Evaluate once on the fixed unseen-speaker test set.

See the Word/PDF report in `docs/` for the complete methodology, architecture,
data audit, error analysis, limitations, and references.

## Limitations

- Greedy decoding has no Hindi language-model constraint.
- The compact encoder has only 4.5 million parameters.
- The final model used 200,000 of the available training rows.
- Performance varies with accent, noise, microphone, and domain.
- The reported metrics do not constitute a demographic fairness audit.

## Authors

Siddharth Roy — Indian Institute of Technology Kharagpur

