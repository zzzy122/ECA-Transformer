## ECA-Transformer: Speaker-Conditioned Transformer for Target Speaker Separation

ECA-Transformer is an end-to-end training framework for Target Speaker Extraction/Separation:

- ECAPA conditioning: use the pretrained ECAPA-TDNN from `SpeechBrain` to extract the target speaker's d-vector as the speaker condition;
- Conditioned Attention: apply dot-product attention in the decoder to align time–frequency features with the speaker embedding;
- A Transformer encoder models temporal context; multiple parallel decoders produce masks that are fused via residual gated fusion for robustness.

Main scripts:
- `speaker_separation_with_progress_transformer.py` (ECA-Transformer main, recommended entry)
- `speaker_separation_with_progress_vf.py` (VoiceFilter baseline and ablations for comparison)


### Highlights
- ECAPA speaker conditioning: freeze `speechbrain/spkrec-ecapa-voxceleb` for fast and stable inference;
- Transformer encoding: `nn.TransformerEncoder` for `[time, feature]` sequences;
- Multi-decoder + residual gated fusion: more expressive and robust mask estimation;
- SpecAugment and waveform-level augmentation: frequency/time masking, background noise, reverberation, time-stretch, and pitch-shift (during training);
- Training tricks: AMP mixed precision, gradient clipping, warmup + cosine annealing, early stopping;
- Metrics: SI-SNR, SDR.


## Model Overview

ECA-Transformer has three major parts:

1) Speaker embedding (frozen): ECAPA-TDNN extracts a d-vector (typically 192-D) from the reference audio as the conditioning vector.

2) Encoder (trainable):
- `TransformerEncoder(input_dim=freq_bins, d_model=256, nhead=8, num_layers=3)`
- Input is the magnitude spectrogram of the mixture, shaped `[batch, time, freq]`.

3) Decoding and fusion (trainable):
- Multiple parallel `SpeakerConditionedDecoder`s: compute dot-product attention `Q(x)·K(emb)` to form temporal attention weights; broadcast speaker embedding to the time axis and concatenate with encoder features; pass through an MLP to output a mask;
- `ResidualGatedFusion` fuses masks from multiple decoders via learned gates and a residual path;
- Apply the fused mask to the magnitude spectrogram and reconstruct the waveform with the mixture phase via ISTFT.


## Environment

Python 3.9+ is recommended. CUDA is highly recommended for training and inference speed.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain tqdm pandas matplotlib pesq pystoi huggingface_hub
```

Notes:
- `pesq` and `pystoi` may require build tools on Windows; if installation is difficult, you can skip them (the evaluation script will fallback).
- Training defaults to GPU if available; otherwise it will run on CPU (much slower).


## Data Preparation

We assume a single-channel dataset organized as one subfolder per speaker (parent folder name serves as speaker ID):

```
root_dataset/
  speaker_A/ a1.wav, a2.wav, ...
  speaker_B/ b1.wav, b2.wav, ...
  ...
```

Duration: by default each clip is cropped or padded to `5s` (`Config.duration`). Audio will be resampled to `16kHz`.

- Build a split quickly (creates `dataset_split.pkl`):
  - In `speaker_separation_with_progress_vf.py`, set:
    - `Config.train_dir1`, `Config.train_dir2` to your data roots (can be the same or different);
    - `Config.output_dir` (default `model_output_5_1`).
  - Run:
    ```bash
    python speaker_separation_with_progress_vf.py
    ```
  - On first run it scans data and saves `output_dir/dataset_split.pkl` (approx. 10:1:1 train/val/test). Training scripts will reuse it afterward.

If you already have a custom split, place `dataset_split.pkl` under the `Config.output_dir` used in `speaker_separation_with_progress_transformer.py`.


## Quick Start

### 1) Train ECA-Transformer (recommended)

`speaker_separation_with_progress_transformer.py` will load the existing split and start training:

```bash
python speaker_separation_with_progress_transformer.py
```

Key config (`Config` in the file):
- Audio/STFT: `sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80`
- Training: `batch_size` (scale with VRAM; on V100 32GB you can go up to ~128), `num_epochs`, `learning_rate=4e-4`, `weight_decay=5e-5`, `warmup_epochs=8`, `patience=5`
- Outputs: `output_dir`, `model_save_path`, `metrics_path`

Defaults during training:
- AMP mixed precision, gradient clipping (5.0), warmup + cosine annealing, early stopping;
- On-the-fly mixture construction: sample target and interferer on the fly in each batch; speaker condition is taken from another utterance of the same speaker;
- Data augmentation (configurable).

Artifacts:
- Best checkpoint: `<output_dir>/best_model.pth`
- Final checkpoint: `<output_dir>/final_model.pth`
- Metrics CSV: `<output_dir>/metrics_comparison.csv`

### Pretrained Weights & Checkpoints

- Download link (as provided):
  - Baidu Netdisk: `https://pan.baidu.com/s/1SUQ1pHxLaDpr0sV-7y3O2g?pwd=ptai` (Extraction code: `ptai`)
  - File name in the share: `ECA-Transformer model.zip` (shared by Baidu Netdisk Super Member v6)

- Contents:
  - ECA-Transformer: `best_model.pth`, `final_model.pth`
  - best_model_voicefilter.pth
  - best_model_ablation.pth

- Where to place:
  - Put ECA-Transformer checkpoints under the `Config.output_dir` of `speaker_separation_with_progress_transformer.py` (default `model_output_3_2`).
  - Put the VoiceFilter d-vector weights at `model_output_5_1/voicefilter_model.pt` (or adjust the path in `speaker_separation_with_progress_vf.py`).

- About size: checkpoints are full training states containing the following keys, which makes them large (total >10GB is normal):
  - `epoch`, `model_state_dict` (weights), `optimizer_state_dict`, `scheduler_state_dict`, `loss`, `si_snr`, `scaler_state_dict` (AMP scaler), `best_val_si_snr`.

- Lightweight loading for inference only:
  ```python
  ckpt = torch.load('model_output_3_2/best_model.pth', map_location=device)
  model.load_state_dict(ckpt['model_state_dict'])
  model.eval()
  ```

### 2) Evaluation / Test

Each epoch is validated, and the final test split is evaluated:
- Metrics: `SI-SNR` and `SDR` (averaged);


### 3) Single-sample Inference Example

```python
import torch, torchaudio
from speaker_separation_with_progress_transformer import ECAPA_TSVS, Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build model and load best weights
model = ECAPA_TSVS(
    sr=Config.sample_rate,
    n_fft=Config.n_fft,
    hop_length=Config.hop_length,
    n_mels=Config.n_mels,
).to(device).eval()

ckpt = torch.load(Config.model_save_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

# Read mixture and reference (target speaker)
mix, sr = torchaudio.load('mixture.wav')
ref, _ = torchaudio.load('reference.wav')

mix = torchaudio.functional.resample(mix, sr, Config.sample_rate)
ref = torchaudio.functional.resample(ref, sr, Config.sample_rate)

mix = mix.mean(0, keepdim=True)  # mono
ref = ref.mean(0, keepdim=True)

mix = mix.to(device)
ref = ref.to(device)

with torch.no_grad():
    enhanced, _, _, _ = model(mix, ref)

torchaudio.save('enhanced.wav', enhanced.cpu().unsqueeze(0), Config.sample_rate)
```


## Loss & Metrics

- Training loss: `L = 0.6 * SI-SNR + 0.4 * spectral/phase-consistency loss` (implemented and switchable);
- Evaluation: `SI-SNR`, `SDR` (default);


## Configuration

- Audio: `sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, duration=5.0`
- Training: `batch_size`, `num_epochs`, `learning_rate=4e-4`, `weight_decay=5e-5`, `warmup_epochs=8`, `patience=5`
- Paths: `output_dir`, `dataset_split_path`, `model_save_path`, `metrics_path`


## Practical Tips

- Hardware: with 24GB+ VRAM you can use larger `batch_size` (e.g., 64–128); reduce it if you run out of memory;
- Reproducibility: fixed `SEED=42` in code;
- Monitoring: watch validation `SI-SNR` and rely on early stopping;
- Artifacts: check `best_model.pth` and `metrics_comparison.csv` under `output_dir`;

## Project Structure

```
ECA-Transformer/
  ├─ speaker_separation_with_progress_transformer.py   # ECA-Transformer main script
  ├─ speaker_separation_with_progress_vf.py            # VoiceFilter baseline / ablations
  └─ speaker_separation_with_progress_ablation.py      # other supplementary experiments
```


## Acknowledgements

- Speaker embeddings use `SpeechBrain ECAPA-TDNN`: `speechbrain/spkrec-ecapa-voxceleb`;
- We also reference the VoiceFilter d-vector concept and open-source implementations.


### References

- A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
- J. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech 2020.
- Q. Wang et al., "VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectral Masking," Interspeech 2019.
- D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," Interspeech 2019.
- Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation," IEEE/ACM TASLP 2019.
- A. W. Rix et al., "Perceptual evaluation of speech quality (PESQ)," IEEE ICASSP 2001.
- C. H. Taal et al., "A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech," IEEE TASL 2011.






