import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict
import warnings
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from pesq import pesq
from pystoi import stoi
import glob
import torchaudio.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import pickle
from huggingface_hub import hf_hub_download
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.activations import Swish
import math


# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试

# 设备配置 - 使用V100 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# 模型参数配置
class Config:
    # 音频参数
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    duration = 5.0  # 固定5秒音频
    
    # 模型参数
    freq_bins = n_fft // 2 + 1
    emb_dim = 256  # d-vector维度
    
    # 训练参数
    batch_size = 128  # V100 32GB内存适合的批次大小
    num_epochs = 200
    learning_rate = 4e-4
    weight_decay = 5e-5  # 增加权重衰减
    patience = 5  # 减少早停耐心值
    warmup_epochs = 8  # warmup阶段epoch数
    
    # 数据加载参数
    num_workers = 24  # 24核CPU
    
    # 数据增强参数
    noise_dataset_path = "noise_dataset"  # 背景噪声数据集路径
    rir_dataset_path = "rir_dataset"  # 房间脉冲响应数据集路径
    
    # 路径配置
    output_dir = "model_output_3_2"
    dataset_split_path = os.path.join(output_dir, "dataset_split.pkl")  # 数据集划分保存路径
    model_save_path = os.path.join(output_dir, "best_model.pth")
    log_path = os.path.join(output_dir, "training_log.json")
    metrics_path = os.path.join(output_dir, "metrics_comparison.csv")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

config = Config()

# 数据增强类
class AudioAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noises = self._load_noises(config.noise_dataset_path)
        self.rirs = self._load_rirs(config.rir_dataset_path)
        
    def _load_noises(self, path):
        """加载背景噪声数据集"""
        noises = []
        if os.path.exists(path):
            for file in glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.flac")):
                try:
                    noise, sr = torchaudio.load(file)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        noise = resampler(noise)
                    noises.append(noise)
                except:
                    continue
        print(f"Loaded {len(noises)} background noises")
        return noises
    
    def _load_rirs(self, path):
        """加载房间脉冲响应"""
        rirs = []
        if os.path.exists(path):
            for file in glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.flac")):
                try:
                    rir, sr = torchaudio.load(file)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        rir = resampler(rir)
                    rirs.append(rir)
                except:
                    continue
        print(f"Loaded {len(rirs)} room impulse responses")
        return rirs
    
    def add_background_noise(self, clean, snr_db=15):
        """添加背景噪声"""
        if not self.noises:
            return clean
        
        noise = random.choice(self.noises)
        
        # 裁剪噪声
        if noise.size(1) < clean.size(0):
            noise = noise.repeat(1, (clean.size(0) // noise.size(1)) + 1)
        noise = noise[:, :clean.size(0)]
        
        # 计算噪声能量
        clean_power = torch.sum(clean ** 2)
        noise_power = torch.sum(noise ** 2)
        
        # 设置信噪比
        snr = 10 ** (snr_db / 10)
        scale = torch.sqrt(clean_power / (snr * noise_power + 1e-8))
        noisy = clean + scale * noise.squeeze(0)
        return noisy / torch.max(torch.abs(noisy))
    
    def add_reverb(self, clean):
        """添加房间混响"""
        if not self.rirs:
            return clean
        
        rir = random.choice(self.rirs)
        
        # 应用卷积
        reverbed = torch.nn.functional.conv1d(
            clean.unsqueeze(0).unsqueeze(0), 
            rir.unsqueeze(0).unsqueeze(0),
            padding=rir.size(1) - 1
        ).squeeze()
        return reverbed / torch.max(torch.abs(reverbed))
    
    def time_stretch(self, waveform, factor=0.9):
        """时间拉伸，保持输出长度不变"""
        original_length = waveform.size(0)
        
        # 计算新长度
        new_length = int(original_length / factor)
        
        # 重采样实现时间拉伸
        stretched = F.resample(
            waveform.unsqueeze(0), 
            orig_freq=self.sample_rate, 
            new_freq=int(self.sample_rate * factor)
        )
        
        # 裁剪或填充到原始长度
        stretched = fix_audio_length(stretched.squeeze(0), original_length)
        
        return stretched
    
    def pitch_shift(self, waveform, semitones=2):
        """音高变换，保持输出长度不变"""
        original_length = waveform.size(0)
        
        # 音高变换因子 (2^(semitones/12))
        factor = 2 ** (semitones / 12)
        
        # 先时间拉伸
        stretched = self.time_stretch(waveform, 1/factor)
        
        # 然后重采样回原始采样率
        shifted = F.resample(
            stretched.unsqueeze(0), 
            orig_freq=int(self.sample_rate / factor), 
            new_freq=self.sample_rate
        )
        
        # 确保长度不变
        shifted = fix_audio_length(shifted.squeeze(0), original_length)
        
        return shifted
    
# 初始化数据增强器
augmenter = AudioAugmenter(sample_rate=config.sample_rate)

# 频谱增强类
class SpecAugment(nn.Module):
    """频谱增强"""
    def __init__(self, freq_mask_param=15, time_mask_param=30, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def forward(self, spec):
        # spec: [batch, time, freq]
        # 频率掩蔽
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, max(1, spec.size(2) - f), (1,)).item()
            spec[:, :, f0:f0+f] = 0
        
        # 时间掩蔽
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, max(1, spec.size(1) - t), (1,)).item()
            spec[:, t0:t0+t, :] = 0
        
        return spec


# 定义我们的模型架构 
class TransformerEncoder(nn.Module):
    """共享的Transformer编码器"""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024, 
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [batch, time, input_dim]
        x = self.linear_in(x)
        x = x.permute(1, 0, 2)  # [time, batch, features]
        x = self.transformer(x)
        return x.permute(1, 0, 2)  # [batch, time, features]
    
class SpeakerConditionedDecoder(nn.Module):
    """带说话人注意力机制的解码器"""
    def __init__(self, input_dim, output_dim, emb_dim=256, hidden_dim=384, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        
        # 注意力机制
        self.attn_query = nn.Linear(input_dim, emb_dim)
        self.attn_key = nn.Linear(emb_dim, emb_dim)
        self.attn_scale =nn.Parameter(torch.tensor(1.0))
        
        # 解码网络
        layers = []
        in_dim = input_dim + emb_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 新增层归一化加速收敛
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
        # 打印调试信息
        print(f"创建 SpeakerConditionedDecoder: input_dim={input_dim}, output_dim={output_dim}, emb_dim={emb_dim}")
        
    def forward(self, x, speaker_emb):
        """
        x: 编码特征 [batch, time, input_dim]
        speaker_emb: 说话人嵌入 [batch, emb_dim]
        """
        # 打印输入形状
        print(f"Decoder输入: x={x.shape}, speaker_emb={speaker_emb.shape}")
        
        # 计算注意力权重
        queries = self.attn_query(x)  # [batch, time, emb_dim]
        keys = self.attn_key(speaker_emb).unsqueeze(1)  # [batch, 1, emb_dim]
        
        # 打印中间形状
        print(f"Decoder中间: queries={queries.shape}, keys={keys.shape}")
        
        # 点积注意力
        attn_scores = torch.matmul(queries, keys.transpose(1, 2))  # [batch, time, 1]
        attn_scores = attn_scores * self.attn_scale
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, time, 1]
        
        # 计算上下文向量
        context = attn_weights * speaker_emb.unsqueeze(1)  # [batch, time, emb_dim]
        
        # 拼接特征和上下文
        decoder_input = torch.cat([x, context], dim=-1)
        
        # 通过解码网络
        mask = self.net(decoder_input)
        return mask, attn_weights

    

class ResidualGatedFusion(nn.Module):
    """残差门控融合模块"""
    def __init__(self, num_decoders, feature_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * num_decoders, num_decoders),
            nn.Softmax(dim=-1)
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, decoder_outputs):
        # decoder_outputs: 解码器输出列表 [batch, time, freq] * 3
        concated = torch.cat(decoder_outputs, dim=-1)  # [batch, time, freq*3]
        gates = self.gate_net(concated)  # [batch, time, 3]
        
        # 应用门控权重
        weighted_sum = sum(gates[..., i].unsqueeze(-1) * decoder_outputs[i] 
                          for i in range(len(decoder_outputs)))
        
        # 残差连接 (使用第一个解码器输出)
        residual = self.residual_scale * decoder_outputs[0]
        return weighted_sum + residual
    

class SpeakerSeparationSystem(nn.Module):
    """带两个并联解码器的分离系统（含说话人注意力）"""
    def __init__(self, freq_bins, emb_dim=256, num_decoders=2):
        super().__init__()
        # 编码器输出维度固定为256
        self.encoder = TransformerEncoder(input_dim=freq_bins, d_model=256, dropout=0.2)
        
        # 创建解码器
        self.decoders = nn.ModuleList([
            SpeakerConditionedDecoder(
                input_dim=256,  # 编码器输出维度
                output_dim=freq_bins,
                emb_dim=emb_dim,  # 使用传入的嵌入维度
                dropout=0.3
            )
            for _ in range(num_decoders)
        ])
        
        # 融合模块
        self.fusion = ResidualGatedFusion(
            num_decoders=num_decoders, 
            feature_dim=freq_bins
        )
        
        # 打印调试信息
        print(f"创建 SpeakerSeparationSystem: freq_bins={freq_bins}, emb_dim={emb_dim}")
        
    def forward(self, mix_spec, speaker_emb):
        
        # 打印输入形状
        original_shape = mix_spec.shape
        print(f"分离系统输入: mix_spec={mix_spec.shape}, speaker_emb={speaker_emb.shape}")
        
        # 编码器处理
        enc_out = self.encoder(mix_spec) # [batch, time, 256]
        
        # 解码器并行处理
        decoder_outputs = []
        attn_weights = []
        for decoder in self.decoders:
            mask, attn = decoder(enc_out, speaker_emb)
            decoder_outputs.append(mask)
            attn_weights.append(attn)
        
        # 残差门控融合
        time_dim = mix_spec.size(1)  # 使用输入的时间维度
        
        fused_mask = self.fusion(decoder_outputs)
        
        
        if fused_mask.shape != original_shape:
            print(f"维度不匹配! 输入: {original_shape}, 输出: {fused_mask.shape}")
            # 自动修正：插值到原始维度
            fused_mask = fused_mask.permute(0, 2, 1)  # [batch, freq, time]
            fused_mask = torch.nn.functional.interpolate(
                fused_mask, 
                size=(original_shape[2], original_shape[1]),  # [freq, time]
                mode='bilinear',
                align_corners=False
            )
            fused_mask = fused_mask.permute(0, 2, 1)
        
        
        # 应用mask获取目标语音
        target_spec = fused_mask * mix_spec
        return target_spec, attn_weights
    

# 修复后的 ECAPA-TDNN
class FixedTDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        activation=nn.SiLU,  # 使用 PyTorch 自带的 SiLU 激活函数
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding="same",
        )
        self.activation = activation()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.activation(self.norm(self.conv(x)))

# 修复后的 ECAPA-TDNN
class FixedECAPA_TDNN(nn.Module):
    def __init__(
        self,
        input_size,
        lin_neurons=192,
        in_channels=1,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        groups=[1, 1, 1, 1, 1],
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.blocks = nn.ModuleList()

        # 初始卷积层
        self.blocks.append(
            FixedTDNNBlock(
                in_channels,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                groups=groups[0],
            )
        )

        # 中间层
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                FixedTDNNBlock(
                    channels[i - 1],
                    channels[i],
                    kernel_sizes[i],
                    dilations[i],
                    groups=groups[i],
                )
            )

        # 最后一层是特殊的，使用点卷积
        self.blocks.append(
            FixedTDNNBlock(
                channels[-2],
                channels[-1],
                kernel_sizes[-1],
                dilations[-1],
                groups=groups[-1],
            )
        )

        # 注意力统计池化
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=True,
        )

        # 使用 PyTorch 自带的 Linear 层
        self.fc = nn.Linear(
            channels[-1] * 2,  # 输入维度
            lin_neurons        # 输出维度
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # 最小化输入长度
        T = x.shape[1]
        x = x.transpose(1, 2)  # (batch, channel, time)
        
        # 通过TDNN层
        for layer in self.blocks:
            x = layer(x)
            
        # 池化
        x = x.transpose(1, 2)  # (batch, time, channel)
        x = self.asp(x)
        
        # 线性层
        x = self.fc(x)
        return x

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        
        if global_context:
            self.tdnn = nn.Linear(channels * 3, attention_channels)
        else:
            self.tdnn = nn.Linear(channels, attention_channels)
            
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        """Computes attentive statistics pooling.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        L = x.shape[1]
        h = x
        
        # 全局上下文
        if self.global_context:
            # 全局均值
            mean = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, channel)
            # 全局标准差
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps)  # (batch, 1, channel)
            # 拼接
            context = torch.cat((x, mean.repeat(1, L, 1), std.repeat(1, L, 1)), dim=2)  # (batch, time, channel*3)
        else:
            context = x

        # 计算注意力权重
        w = self.tdnn(context)  # (batch, time, attention_channels)
        w = self.tanh(w)
        w = self.conv(w.transpose(1, 2))  # (batch, channel, time)
        w = F.softmax(w, dim=2)  # (batch, channel, time)

        # 计算加权统计量
        mu = torch.sum(x * w.transpose(1, 2), dim=1)  # (batch, channel)
        rh = torch.sqrt(
            (torch.sum((x**2) * w.transpose(1, 2), dim=1) - mu**2)
            .clamp(min=self.eps)
        )  # (batch, channel)

        # 拼接均值和标准差
        x = torch.cat((mu, rh), dim=1)  # (batch, channel*2)
        return x
    
    

class ECAPA_TSVS(nn.Module):
    """完整的说话人分离验证系统（带注意力机制和SpecAugment）"""
    def __init__(self, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1  # 513
        
        # 语音特征处理
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # 加载预训练 ECAPA-TDNN 模型
        print("加载预训练的 ECAPA-TDNN 模型...")
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir="pretrained_models/ecapa"
        )
        
        # 获取嵌入维度
        try:
            # 直接获取嵌入维度
            test_input = torch.randn(1, 16000).to(device)
            with torch.no_grad():
                test_emb = self.ecapa.encode_batch(test_input)
                self.emb_dim = test_emb.shape[-1]
        except Exception as e:
            self.emb_dim = 192  # ECAPA-TDNN 的标准输出维度
        
        # 冻结 ECAPA 模型参数
        print("冻结 ECAPA 模型参数...")
        self.ecapa.eval()
        for param in self.ecapa.parameters():
            param.requires_grad = False
            
        # 分离系统 - 使用实际的嵌入维度
        self.separation = SpeakerSeparationSystem(
            freq_bins=self.freq_bins,
            emb_dim=self.emb_dim
        )
        
        # SpecAugment
        self.spec_augment = SpecAugment(
            freq_mask_param=20,
            time_mask_param=40,
            num_freq_masks=2,
            num_time_masks=2
        )
        
        # 打印模型结构
        print("分离系统结构:")
        print(self.separation)
        
    def forward(self, mixture, reference):
        """
        mixture: 混合语音 [batch, samples]
        reference: 参考语音 [batch, samples]
        """
        # 1. 混合语音 STFT 处理
        window = torch.hann_window(self.n_fft).to(mixture.device)
        mix_stft = torch.stft(
            mixture, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        mix_mag = torch.abs(mix_stft)  # 幅度谱 [batch, freq, time]
        
        # 2. 参考语音通过 ECAPA-TDNN
        with torch.no_grad():
            speaker_emb = self.ecapa.encode_batch(reference).squeeze(1)
        
        # 3. 确保幅度谱形状正确 [batch, time, freq]
        mix_mag = mix_mag.permute(0, 2, 1)  # 从 [batch, freq, time] 转为 [batch, time, freq]
        
        # 4. 训练时应用 SpecAugment
        if self.training and random.random() < 0.5:
            mix_mag = self.spec_augment(mix_mag)
        
        # 5. 分离系统处理
        target_mag, attn_weights = self.separation(mix_mag, speaker_emb)
        
        # 6. 使用混合语音相位重建时域信号
        # 恢复相位谱的原始形状 [batch, freq, time]
        phase = torch.angle(mix_stft)
        # 目标幅度谱形状是 [batch, time, freq]，需要转为 [batch, freq, time]
        target_mag = target_mag.permute(0, 2, 1)
        
        print(f"分离前: {mix_mag.shape}, 分离后: {target_mag.shape},原始: {phase.shape}")
        
        # 重建复数谱
        target_stft = target_mag * torch.exp(1j * phase)
        
        if target_stft.size(-1) != phase.size(-1):
            print(f"维度不匹配: target_stft={target_stft.shape}, phase={phase.shape}")
            target_stft = torch.nn.functional.interpolate(target_stft, size=phase.size(-1), mode='linear', align_corners=False)
        
        target_audio = torch.istft(
            target_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=window,
            length=mixture.size(1)
        )
        
        return target_audio, target_mag, attn_weights, speaker_emb
    

def fix_audio_length(waveform, target_length):
    """确保音频长度固定"""
    current_length = waveform.size(0)
    
    if current_length > target_length:
        # 随机裁剪
        start = torch.randint(0, current_length - target_length, (1,)).item()
        return waveform[start:start+target_length]
    elif current_length < target_length:
        # 填充
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding), mode='constant')
    else:
        return waveform

class SpeakerSeparationDataset(Dataset):
    def __init__(self, file_list, sample_rate=16000, duration=4.0):
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_clip = int(sample_rate * duration)
        print(f"Loaded dataset with {len(file_list)} audio files")
    
        self.speaker_ids = []
        for file_path in self.file_list:
            # 从文件路径中提取说话人ID (倒数第二级目录)
            path_parts = file_path.split(os.sep)
            if len(path_parts) < 2:
                speaker_id = "unknown"
            else:
                speaker_id = path_parts[-2]  # 倒数第二级目录是说话人ID
            self.speaker_ids.append(speaker_id)
        
        # 创建说话人到文件索引的映射
        self.speaker_to_indices = defaultdict(list)
        for idx, speaker_id in enumerate(self.speaker_ids):
            self.speaker_to_indices[speaker_id].append(idx)
        
        print(f"Loaded dataset with {len(file_list)} audio files, {len(self.speaker_to_indices)} speakers")
    
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        speaker_id = self.speaker_ids[idx]

        # 加载音频
        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            # 如果加载失败，返回随机音频
            waveform = torch.randn(1, self.samples_per_clip)
            sr = self.sample_rate
        
        # 确保采样率一致
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 确保单声道
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 确保长度一致
        waveform = waveform.squeeze(0)
        if waveform.size(0) != self.samples_per_clip:
            waveform = fix_audio_length(waveform, self.samples_per_clip)
        
        return waveform, audio_path,speaker_id 


# 语音质量评估函数
def calculate_audio_metrics(clean, enhanced, sr=16000):
    """计算语音质量评估指标"""
    metrics = {}

    # SDR (信噪比)
    clean = clean.squeeze()
    enhanced = enhanced.squeeze()
    min_len = min(clean.shape[0], enhanced.shape[0])
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # 1. SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    def si_snr(estimated, target):
        # 移除直流分量
        target = target - torch.mean(target)
        estimated = estimated - torch.mean(estimated)
        
        # 计算最优缩放因子
        dot = torch.sum(estimated * target)
        target_energy = torch.sum(target ** 2)
        scale = dot / (target_energy + 1e-8)
        
        # 计算目标信号和噪声
        target_scaled = scale * target
        noise = estimated - target_scaled
        
        # 计算SI-SNR
        si_snr_val = 10 * torch.log10(
            (torch.sum(target_scaled ** 2) + 1e-8) / 
            (torch.sum(noise ** 2) + 1e-8)
        )
        return si_snr_val.item()
    
    # 2. SDR (Signal-to-Distortion Ratio)
    def sdr(estimated, target):
        target = target - torch.mean(target)
        estimated = estimated - torch.mean(estimated)
        noise = estimated - target
        target_energy = torch.sum(target **2)
        noise_energy = torch.sum(noise** 2)
        sdr_val = 10 * torch.log10((target_energy + 1e-8) / (noise_energy + 1e-8))
        return sdr_val.item()
    
    metrics['SI-SNR'] = si_snr(enhanced, clean)
    metrics['SDR'] = sdr(enhanced, clean)
    
    
    return metrics

# 创建混合样本 (带数据增强)
def create_mixed_sample(dataset, index):
    """为训练创建混合样本，应用数据增强"""
    fixed_length = int(config.sample_rate * config.duration)
    
    # 获取目标说话人样本
    target_waveform, target_path, target_speaker = dataset[index]
    
    # 随机选择干扰说话人样本 (确保不同说话人)
    while True:
        other_idx = random.randint(0, len(dataset) - 1)
        _, _, other_speaker = dataset[other_idx]
        if other_speaker != target_speaker:
            break
    
    interference_waveform, _, _ = dataset[other_idx]
    
    # 随机混合比例 (SNR在0-5dB之间)
    snr_db = random.uniform(0, 5)
    target_energy = torch.sum(target_waveform ** 2)
    interference_energy = torch.sum(interference_waveform ** 2)
    
    # 计算缩放因子
    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(target_energy / (snr * interference_energy + 1e-8))
    
    # 创建混合语音
    mixed_waveform = target_waveform + scale * interference_waveform
    
    # 数据增强 (仅在训练模式)
    if dataset.mode == 'train':
        if random.random() < 0:  # 70%概率添加背景噪声
            mixed_waveform = augmenter.add_background_noise(
                mixed_waveform, 
                snr_db=random.uniform(10, 25)  # 10-25dB SNR
            )
        
        if random.random() < 0:  # 40%概率添加混响
            mixed_waveform = augmenter.add_reverb(mixed_waveform)
        
        if random.random() < 0:  # 30%概率时间拉伸
            factor = random.uniform(0.85, 1.15)
            mixed_waveform = augmenter.time_stretch(mixed_waveform, factor)
        
        if random.random() < 0:  # 30%概率音高变换
            semitones = random.uniform(-3, 3)
            mixed_waveform = augmenter.pitch_shift(mixed_waveform, semitones)
    
    # 归一化混合语音
    mixed_waveform = mixed_waveform / torch.max(torch.abs(mixed_waveform))
    
    # 获取目标说话人的参考语音 (随机选择同一说话人的另一个样本)
    same_speaker_indices = [i for i in range(len(dataset)) 
                            if dataset.speaker_ids[i] == target_speaker and i != index]
    
    if same_speaker_indices:
        ref_idx = random.choice(same_speaker_indices)
        ref_waveform, ref_path, _ = dataset[ref_idx]
    else:
        # 如果没有其他样本，使用当前样本作为参考
        ref_waveform = target_waveform.clone()
        ref_path = target_path
    
    # 确保所有音频都是固定长度
    target_waveform = fix_audio_length(target_waveform, fixed_length)
    mixed_waveform = fix_audio_length(mixed_waveform, fixed_length)
    ref_waveform = fix_audio_length(ref_waveform, fixed_length)
    
    return mixed_waveform, ref_waveform, target_waveform, target_path

# 自定义数据加载器
class MixedDataLoader(DataLoader):
    """动态生成混合样本的数据加载器 (修复dataset问题)"""
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=12):
        super().__init__(
            dataset, 
            batch_size=None,  # 禁用默认批处理
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,  # 新增：加速CPU到GPU的数据传输
            prefetch_factor=2,  # 新增：每个worker预加载2个batch
            persistent_workers=True  # 新增：保持worker进程不关闭
        )
        # 使用 _batch_size 避免与父类属性冲突
        self._batch_size = batch_size
        
        # 使用父类存储的dataset
        self.indices = list(range(len(dataset)))
        
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        # 使用 _batch_size 计算长度
        return len(self.dataset) // self._batch_size
    
    def collate_fn(self, idx_list):
        batch_mixed = []
        batch_ref = []
        batch_target = []
        batch_paths = []
        
        # 使用父类存储的dataset
        for idx in idx_list:
            mixed, ref, target, path = create_mixed_sample(self.dataset, idx)
            batch_mixed.append(mixed)
            batch_ref.append(ref)
            batch_target.append(target)
            batch_paths.append(path)
        
        return (
            torch.stack(batch_mixed),
            torch.stack(batch_ref),
            torch.stack(batch_target),
            batch_paths
        )
    
    def __iter__(self):
        # 创建索引批次
        batch_indices = []
        for idx in self.indices:
            batch_indices.append(idx)
            # 使用 _batch_size 判断批次大小
            if len(batch_indices) == self._batch_size:
                yield self.collate_fn(batch_indices)
                batch_indices = []
        
        # 处理剩余样本
        if batch_indices:
            yield self.collate_fn(batch_indices)

# 损失函数 - SI-SNR/sdr
def si_snr(estimated, target, eps=1e-8):
    """计算尺度不变信噪比 (SI-SNR)"""
    # 归一化
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
    
    # 计算目标能量
    s_target = target
    s_estimate = estimated
    
    # 计算点积
    dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + eps
    
    # 投影
    proj = dot * s_target / s_target_energy
    
    # 噪声项
    e_noise = s_estimate - proj
    
    # 计算SI-SNR
    target_energy = torch.sum(proj ** 2, dim=-1)
    noise_energy = torch.sum(e_noise ** 2, dim=-1)
    si_snr = 10 * torch.log10((target_energy) / (noise_energy + eps) + eps)
    return -torch.mean(si_snr)


# 频谱和相位
# def spectral_loss(estimated_mag, target_mag):
#     """计算幅度谱损失 (L1损失)"""
#     return torch.mean(torch.abs(estimated_mag - target_mag))

def phase_loss(enhanced_stft, target_stft):
    """相位一致性损失"""
    enhanced_phase = torch.angle(enhanced_stft)
    target_phase = torch.angle(target_stft)
    return torch.mean(torch.abs(torch.sin(enhanced_phase - target_phase)))

# 学习率调度器 (Warmup + 余弦退火)
def create_scheduler(optimizer, num_warmup_epochs, num_training_epochs, initial_lr):
    """创建warmup + 余弦退火学习率调度器"""
    # Warmup阶段
    def warmup_lr_scheduler(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch) / float(max(1, num_warmup_epochs))
        return 1.0
    
    # 余弦退火阶段
    warmup = LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)
    cosine_annealing = CosineAnnealingLR(
        optimizer, 
        T_max=num_training_epochs - num_warmup_epochs, 
        eta_min=initial_lr * 0.01  # 最小学习率为初始值的1%
    )
    
    # 组合调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine_annealing],
        milestones=[num_warmup_epochs]
    )
    
    return scheduler

# 训练函数
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, patience, start_epoch=0, best_val_si_snr=-float('inf')):
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_si_snr': [], 'lr': []}
    
    # 初始化指标比较结果
    metrics_comparison = []
    
    # 混合精度训练的梯度缩放器
    scaler = GradScaler()
    
    for epoch in range(start_epoch, num_epochs):
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.8f}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_si_snr = 0.0
        print(f"Model device: {next(model.parameters()).device}")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for mixed, ref, target, _ in progress_bar:
            mixed = mixed.to(device)
            # print(mixed.device)
            ref = ref.to(device)
            target = target.to(device)
            print(f"mixed device: {mixed.device}, ref device: {ref.device}, target device: {target.device}")
            
            # 混合精度训练
            with autocast():
                # 前向传播
                target_audio, target_mag, _, _ = model(mixed, ref)
                
                # 计算损失
                si_snr_loss = si_snr(target_audio, target)
                
                # 获取目标幅度谱
                with torch.no_grad():
                    target_stft = torch.stft(
                        target, 
                        n_fft=config.n_fft, 
                        hop_length=config.hop_length,
                        return_complex=True
                    )
                    target_mag_gt = torch.abs(target_stft)
                
                # spec_loss = spectral_loss(target_mag, target_mag_gt)
                phase_loss = phase_loss(target_stft, target_mag_gt)
                
                # 混合损失
                loss = 0.6 * si_snr_loss + 0.4 * phase_loss
                
            
            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新统计信息
            train_loss += loss.item()
            train_si_snr += -si_snr_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'SI-SNR': f"{-si_snr_loss.item():.2f} dB"
            })
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_si_snr /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        # val_loss, val_si_snr, epoch_metrics = validate(model, val_loader, voicefilter)
        val_loss, val_si_snr, epoch_metrics = validate(model, val_loader)
        history['val_loss'].append(val_loss)
        history['val_si_snr'].append(val_si_snr)
        
        # 保存指标比较结果
        epoch_metrics['epoch'] = epoch + 1
        metrics_comparison.append(epoch_metrics)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_si_snr > best_val_si_snr:
            best_val_si_snr = val_si_snr
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'si_snr': val_si_snr,
                'scaler_state_dict': scaler.state_dict(),
                'best_val_si_snr': best_val_si_snr
            }, config.model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    return history, metrics_comparison, best_val_si_snr

# def validate(model, val_loader, voicefilter, num_samples=50): 
def validate(model, val_loader, num_samples=50):
    model.eval()
    val_loss = 0.0
    val_si_snr = 0.0
    count = 0
    
    # 初始化指标统计
    our_model_metrics = {'SI-SNR': 0.0, 'PESQ': 0.0, 'STOI': 0.0, 'SDR': 0.0}
    # voicefilter_metrics = {'SI-SNR': 0.0, 'PESQ': 0.0, 'STOI': 0.0, 'SDR': 0.0}
    
    # 用于保存比较结果的样本
    comparison_samples = []
    
    with torch.no_grad():
        for mixed, ref, target, paths in tqdm(val_loader, desc="Validating"):
            mixed = mixed.to(device)
            ref = ref.to(device)
            target = target.to(device)
            
            # 前向传播
            target_audio, target_mag, _, speaker_emb = model(mixed, ref)
            
            # 计算损失
            si_snr_loss = si_snr(target_audio, target)
            
            # 获取目标幅度谱
            target_stft = torch.stft(
                target, 
                n_fft=config.n_fft, 
                hop_length=config.hop_length,
                return_complex=True
            )
            target_mag_gt = torch.abs(target_stft)
            
            spec_loss = spectral_loss(target_mag, target_mag_gt)
            
            # 混合损失
            loss = 0.6 * si_snr_loss + 0.4 * spec_loss 
            
            # 更新统计信息
            val_loss += loss.item()
            val_si_snr += -si_snr_loss.item()
            count += 1
            
            # 随机选择一些样本进行详细评估
            if len(comparison_samples) < num_samples:
                idx = random.randint(0, mixed.size(0)-1)
                sample = {
                    'mixed': mixed[idx].cpu(),
                    'ref': ref[idx].cpu(),
                    'target': target[idx].cpu(),
                    'enhanced': target_audio[idx].cpu(),
                    'speaker_emb': speaker_emb[idx].cpu(),
                    'path': paths[idx]
                }
                comparison_samples.append(sample)
    
    val_loss /= count
    val_si_snr /= count
    
    # 计算详细指标
    for sample in tqdm(comparison_samples, desc="Calculating metrics"):
        # 我们的模型指标
        our_metrics = calculate_audio_metrics(sample['target'], sample['enhanced'], config.sample_rate)
        for k in our_model_metrics:
            our_model_metrics[k] += our_metrics[k]
        

    # 创建指标比较字典
    epoch_metrics = {
        'our_model_SDR': our_model_metrics['SDR']
    }
    
    return val_loss, val_si_snr, epoch_metrics

# 数据加载和预处理 (直接加载现有划分)
def load_and_prepare_data():
    """直接从文件加载数据集划分"""
    if not os.path.exists(config.dataset_split_path):
        raise FileNotFoundError(f"Dataset split file not found: {config.dataset_split_path}")
    
    print("Loading dataset split from file...")
    with open(config.dataset_split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    # 创建数据集对象
    train_dataset = SpeakerSeparationDataset(
        split_data['train'], 
        sample_rate=config.sample_rate,
        duration=config.duration
    )
    
    val_dataset = SpeakerSeparationDataset(
        split_data['val'], 
        sample_rate=config.sample_rate,
        duration=config.duration
    )
    
    test_dataset = SpeakerSeparationDataset(
        split_data['test'], 
        sample_rate=config.sample_rate,
        duration=config.duration
    )
    
    print(f"Dataset loaded: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

# 主函数
def main():
    # 加载数据
    torch.set_grad_enabled(True)
    
    print("Loading data from existing split...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_data()
    
    # 创建数据加载器
    train_loader = MixedDataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = MixedDataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = MixedDataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 初始化模型
    print("Initializing model...")
    model = ECAPA_TSVS(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # 优化器 (增加权重衰减)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98) 
    )
    
    # 学习率调度器 (Warmup + 余弦退火)
    scheduler = create_scheduler(
        optimizer, 
        num_warmup_epochs=config.warmup_epochs,
        num_training_epochs=config.num_epochs,
        initial_lr=config.learning_rate
    )
    
    # 检查是否存在检查点
    start_epoch = 0
    best_val_si_snr = -float('inf')
    if os.path.exists(config.model_save_path):
        print("Loading checkpoint for resuming training...")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_si_snr = checkpoint['best_val_si_snr']
        print(f"Resuming from epoch {start_epoch}, best_val_si_snr: {best_val_si_snr:.2f} dB")
    
    # 训练模型
    print("Starting training...")
    history, metrics_comparison, best_val_si_snr = train(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler,
        num_epochs=config.num_epochs,
        patience=config.patience,
        start_epoch=start_epoch,
        best_val_si_snr=best_val_si_snr
    )
    
    # 测试模型
    print("Testing model...")
    test_loss, test_si_snr, test_metrics = validate(
        model, 
        test_loader
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test SI-SNR: {test_si_snr:.2f} dB")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, os.path.join(config.output_dir, "final_model.pth"))
    
    # 保存测试结果
    test_metrics['epoch'] = 'Test'
    metrics_comparison.append(test_metrics)
    pd.DataFrame(metrics_comparison).to_csv(config.metrics_path, index=False)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
    

    
    
    
    
# 数据处理
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict
import warnings
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from pesq import pesq
from pystoi import stoi
import glob
import torchaudio.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import pickle 

def load_and_prepare_data():
    """从两个文件夹加载数据并准备120,000条数据，保存划分结果"""
    # 如果已有保存的划分，直接加载
    if os.path.exists(config.dataset_split_path):
        print("Loading dataset split from file...")
        with open(config.dataset_split_path, 'rb') as f:
            split_data = pickle.load(f)
            return split_data['train'], split_data['val'], split_data['test']
    
    # 收集所有音频文件
    all_audio_files = []
    
    # 处理第一个文件夹
    print(f"Scanning {config.train_dir1} for audio files...")
    for root, _, files in os.walk(config.train_dir1):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                file_path = os.path.join(root, file)
                # 检查音频长度
                try:
                    info = torchaudio.info(file_path)
                    duration = info.num_frames / info.sample_rate
                    if duration >= 3.0:  # 只保留长度大于3秒的音频
                        all_audio_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    # 处理第二个文件夹
    print(f"Scanning {config.train_dir2} for audio files...")
    for root, _, files in os.walk(config.train_dir2):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                file_path = os.path.join(root, file)
                # 检查音频长度
                try:
                    info = torchaudio.info(file_path)
                    duration = info.num_frames / info.sample_rate
                    if duration >= 3.0:  # 只保留长度大于3秒的音频
                        all_audio_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    print(f"Found {len(all_audio_files)} audio files with duration >=3 seconds")
    
    # 确保有足够的数据
    if len(all_audio_files) < 120000:
        raise ValueError(f"Only {len(all_audio_files)} files found, but 120,000 are required.")
    
    # 按用户ID分组
    speaker_data = defaultdict(list)
    for file_path in all_audio_files:
        speaker_id = os.path.basename(os.path.dirname(file_path))
        speaker_data[speaker_id].append(file_path)
    
    print(f"Found {len(speaker_data)} unique speakers")
    
    # 按用户ID划分数据集 (10:1:1)
    speaker_ids = list(speaker_data.keys())
    random.shuffle(speaker_ids)
    
    # 计算所需文件数
    total_files_needed = 120000
    train_files_needed = 100000
    val_files_needed = 10000
    test_files_needed = 10000
    
    # 分配用户到不同数据集
    train_files = []
    val_files = []
    test_files = []
    
    # 先分配训练集用户
    for sid in speaker_ids:
        # 如果训练集还未满，添加该用户的所有音频
        if len(train_files) < train_files_needed:
            # 计算当前用户可添加的最大文件数（不超过训练集剩余容量）
            max_to_add = min(len(speaker_data[sid]), train_files_needed - len(train_files))
            if max_to_add > 0:
                selected_files = random.sample(speaker_data[sid], max_to_add)
                train_files.extend(selected_files)
        
        # 然后分配验证集
        elif len(val_files) < val_files_needed:
            max_to_add = min(len(speaker_data[sid]), val_files_needed - len(val_files))
            if max_to_add > 0:
                selected_files = random.sample(speaker_data[sid], max_to_add)
                val_files.extend(selected_files)
        
        # 最后分配测试集
        elif len(test_files) < test_files_needed:
            max_to_add = min(len(speaker_data[sid]), test_files_needed - len(test_files))
            if max_to_add > 0:
                selected_files = random.sample(speaker_data[sid], max_to_add)
                test_files.extend(selected_files)
        
        # 如果所有数据集都已满足，停止分配
        if (len(train_files) >= train_files_needed and 
            len(val_files) >= val_files_needed and 
            len(test_files) >= test_files_needed):
            break
    
    # 确保精确数量
    train_files = train_files[:train_files_needed]
    val_files = val_files[:val_files_needed]
    test_files = test_files[:test_files_needed]
    
    print(f"After initial assignment: Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")
    
    # 检查用户ID是否跨数据集
    def get_speaker_ids(files):
        return {os.path.basename(os.path.dirname(f)) for f in files}
    
    train_speakers = get_speaker_ids(train_files)
    val_speakers = get_speaker_ids(val_files)
    test_speakers = get_speaker_ids(test_files)
    
    # 验证没有重叠的说话人
    assert len(train_speakers & val_speakers) == 0, "训练集和验证集有重叠用户"
    assert len(train_speakers & test_speakers) == 0, "训练集和测试集有重叠用户"
    assert len(val_speakers & test_speakers) == 0, "验证集和测试集有重叠用户"
    
    print(f"Final dataset split: Train {len(train_files)} files ({len(train_speakers)} speakers), "
          f"Val {len(val_files)} files ({len(val_speakers)} speakers), "
          f"Test {len(test_files)} files ({len(test_speakers)} speakers)")
    
    # 保存数据集划分
    split_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    with open(config.dataset_split_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"Dataset split saved to {config.dataset_split_path}")
    
    return train_files, val_files, test_files

# 在Config类中添加训练目录配置
class Config:
    
    # 添加训练目录
    train_dir1 = "data_train_1"  # 替换为实际路径
    train_dir2 = "data_train"  # 替换为实际路径
    dataset_split_path = os.path.join('model_output_3_2, "dataset_split.pkl")

config = Config() 


train_files, val_files, test_files = load_and_prepare_data()