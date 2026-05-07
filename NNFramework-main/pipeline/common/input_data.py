"""部署侧公用的输入配置、音频数据集与 log-mel 特征工具。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import wave

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .frontend_shape_utils import infer_feature_frame_count


# 当前部署链路只直接支持 PCM WAV 输入，避免把 pipeline 重新耦合回训练脚本。
SUPPORTED_WAV_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class WaveInfo:
    """单个音频文件的基础元数据。"""

    path: Path
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int


@dataclass(frozen=True)
class SegmentRecord:
    """切窗后单个样本片段的定位信息。"""

    path: str
    label_index: int
    split: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class SplitBundle:
    """按文件先划分、再切窗后的 train/val/test 索引。"""

    train: list[SegmentRecord]
    val: list[SegmentRecord]
    test: list[SegmentRecord]
    file_manifest: dict[str, dict[str, list[str]]]


def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并保证根节点是字典。"""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to load the deployment config") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def resolve_config_data_root(config_path: Path, raw_value: str) -> Path:
    """把 YAML 中的 data_root 解析成真实绝对路径。"""

    raw_path = Path(str(raw_value))
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        (config_path.parent / raw_path).resolve(),
        (config_path.parent.parent / raw_path).resolve(),
    ]
    if raw_path.parts and raw_path.parts[0].lower() == "in" and len(raw_path.parts) > 1:
        candidates.append((config_path.parent / Path(*raw_path.parts[1:])).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_input_config(config: dict[str, Any]) -> None:
    """校验部署链真正依赖的配置字段。"""

    classes = config.get("data", {}).get("classes")
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(item, str) and item for item in classes):
        raise ValueError("data.classes must be a non-empty string list with at least 2 classes")

    paths_cfg = config.get("paths", {})
    data_root = paths_cfg.get("data_root")
    if not data_root:
        raise ValueError("paths.data_root is required")

    audio_cfg = config.get("audio", {})
    feature_cfg = config.get("feature", {})
    split_cfg = config.get("split", {})

    required_numeric_fields = [
        ("audio.sample_rate", audio_cfg.get("sample_rate")),
        ("audio.window_sec", audio_cfg.get("window_sec")),
        ("audio.hop_sec", audio_cfg.get("hop_sec")),
        ("feature.n_fft", feature_cfg.get("n_fft")),
        ("feature.win_length", feature_cfg.get("win_length")),
        ("feature.hop_length", feature_cfg.get("hop_length")),
        ("feature.n_mels", feature_cfg.get("n_mels")),
        ("feature.fmin", feature_cfg.get("fmin")),
        ("feature.fmax", feature_cfg.get("fmax")),
    ]
    missing = [name for name, value in required_numeric_fields if value is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if float(audio_cfg["window_sec"]) <= 0 or float(audio_cfg["hop_sec"]) <= 0:
        raise ValueError("audio.window_sec and audio.hop_sec must be > 0")
    if int(audio_cfg["sample_rate"]) <= 0:
        raise ValueError("audio.sample_rate must be > 0")
    if int(feature_cfg["win_length"]) <= 0 or int(feature_cfg["hop_length"]) <= 0 or int(feature_cfg["n_fft"]) <= 0:
        raise ValueError("feature.n_fft / win_length / hop_length must be > 0")
    if int(feature_cfg["n_fft"]) < int(feature_cfg["win_length"]):
        raise ValueError("feature.n_fft must be >= feature.win_length")
    if int(feature_cfg["n_mels"]) <= 0:
        raise ValueError("feature.n_mels must be > 0")

    val_ratio = float(split_cfg.get("val_ratio", 0.0))
    test_ratio = float(split_cfg.get("test_ratio", 0.0))
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("split.val_ratio and split.test_ratio must be >= 0")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("split.val_ratio + split.test_ratio must be < 1.0")

    # 当前部署链路会把 bandpass 视为恢复契约元数据，不再作为入口硬错误拦截。


def load_input_config(config_path: Path) -> dict[str, Any]:
    """读取并规范化部署输入配置。"""

    resolved_config_path = config_path.resolve()
    config = read_yaml(resolved_config_path)
    paths_cfg = config.setdefault("paths", {})
    data_root = paths_cfg.get("data_root")
    if data_root:
        paths_cfg["data_root"] = str(resolve_config_data_root(resolved_config_path, str(data_root)))
    validate_input_config(config)
    return config


def set_random_seed(seed: int) -> None:
    """统一设置随机种子，保证切窗与校准样本选择可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_wave_info(path: Path) -> WaveInfo:
    """读取 WAV 头信息，不解码完整数据。"""

    with wave.open(str(path), "rb") as handle:
        return WaveInfo(
            path=path,
            sample_rate_hz=handle.getframerate(),
            channels=handle.getnchannels(),
            sample_width_bytes=handle.getsampwidth(),
            frame_count=handle.getnframes(),
        )


def decode_pcm_frames(raw_bytes: bytes, sample_width_bytes: int) -> np.ndarray:
    """把常见 PCM 深度解码成 [-1, 1] 浮点波形。"""

    if sample_width_bytes == 1:
        array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        return (array - 128.0) / 128.0
    if sample_width_bytes == 2:
        return np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width_bytes == 3:
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        values = (
            byte_array[:, 0].astype(np.int32)
            | (byte_array[:, 1].astype(np.int32) << 8)
            | (byte_array[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        values = (values ^ sign_mask) - sign_mask
        return values.astype(np.float32) / float(1 << 23)
    if sample_width_bytes == 4:
        return np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / float(1 << 31)
    raise ValueError(f"Unsupported WAV sample width: {sample_width_bytes} bytes")


def load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    """把 WAV 读成单声道浮点张量。"""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width_bytes = handle.getsampwidth()
        sample_rate_hz = handle.getframerate()
        frame_count = handle.getnframes()
        raw_bytes = handle.readframes(frame_count)

    waveform = decode_pcm_frames(raw_bytes, sample_width_bytes)
    waveform = waveform.reshape(-1, channels)
    mono = waveform.mean(axis=1, dtype=np.float32)
    return torch.from_numpy(mono.copy()), sample_rate_hz


def resample_waveform(waveform: torch.Tensor, source_rate_hz: int, target_rate_hz: int) -> torch.Tensor:
    """使用线性插值把波形重采样到目标采样率。"""

    if source_rate_hz == target_rate_hz:
        return waveform
    target_length = max(1, int(round(waveform.numel() * float(target_rate_hz) / float(source_rate_hz))))
    shaped = waveform.view(1, 1, -1)
    resized = F.interpolate(shaped, size=target_length, mode="linear", align_corners=False)
    return resized.view(-1)


def collect_class_files(data_root: Path, classes: list[str], allowed_extensions: list[str]) -> dict[str, list[Path]]:
    """收集每个类别目录下允许的音频文件。"""

    normalized_extensions = {suffix.lower() for suffix in allowed_extensions}
    class_files: dict[str, list[Path]] = {}
    for class_name in classes:
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_extensions
        )
        if not files:
            raise FileNotFoundError(f"No matched audio files in class directory: {class_dir}")
        unsupported = sorted({path.suffix.lower() for path in files if path.suffix.lower() not in SUPPORTED_WAV_SUFFIXES})
        if unsupported:
            raise RuntimeError(
                "Current deployment data loader only supports PCM WAV input. "
                f"Please convert or remove these extensions: {unsupported}"
            )
        class_files[class_name] = files
    return class_files


def derive_split_counts(total_files: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    """按文件数推导 val/test 划分数量。"""

    if total_files <= 1:
        return 0, 0

    test_count = int(total_files * test_ratio)
    val_count = int(total_files * val_ratio)

    if test_ratio > 0 and total_files >= 3 and test_count == 0:
        test_count = 1
    if val_ratio > 0 and (total_files - test_count) >= 2 and val_count == 0:
        val_count = 1

    while test_count + val_count >= total_files:
        if val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return val_count, test_count


def infer_resampled_frame_count(frame_count: int, source_rate_hz: int, target_rate_hz: int) -> int:
    """根据采样率变化估计重采样后的样本数。"""

    if source_rate_hz == target_rate_hz:
        return frame_count
    return max(1, int(round(frame_count * float(target_rate_hz) / float(source_rate_hz))))


def enumerate_segment_starts(total_samples: int, window_samples: int, hop_samples: int, pad_short: bool) -> list[int]:
    """枚举切窗起点，保证与部署链使用的时间窗口规则一致。"""

    if total_samples <= window_samples:
        return [0] if pad_short or total_samples == window_samples else []

    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    if pad_short:
        tail_start = max(total_samples - window_samples, 0)
        if starts[-1] != tail_start:
            starts.append(tail_start)
    return starts


def build_split_bundle(config: dict[str, Any]) -> SplitBundle:
    """从固定数据目录中构建切窗后的 train/val/test 索引。"""

    seed = int(config.get("seed", 42))
    data_root = Path(config["paths"]["data_root"]).resolve()
    classes = list(config["data"]["classes"])
    allowed_extensions = list(config["data"].get("audio_extensions", [".wav"]))
    sample_rate_hz = int(config["audio"]["sample_rate"])
    window_samples = int(round(sample_rate_hz * float(config["audio"]["window_sec"])))
    hop_samples = int(round(sample_rate_hz * float(config["audio"]["hop_sec"])))
    pad_short = bool(config["audio"].get("pad_short", True))
    val_ratio = float(config.get("split", {}).get("val_ratio", 0.0))
    test_ratio = float(config.get("split", {}).get("test_ratio", 0.0))

    class_files = collect_class_files(data_root, classes, allowed_extensions)
    split_records = {"train": [], "val": [], "test": []}
    file_manifest = {"train": {}, "val": {}, "test": {}}
    for class_index, class_name in enumerate(classes):
        files = list(class_files[class_name])
        rng = random.Random(seed + class_index)
        rng.shuffle(files)

        val_count, test_count = derive_split_counts(len(files), val_ratio, test_ratio)
        test_files = files[:test_count]
        val_files = files[test_count:test_count + val_count]
        train_files = files[test_count + val_count:]
        if not train_files:
            raise RuntimeError(f"Class {class_name} has no training files after split")

        file_manifest["train"][class_name] = [str(path) for path in sorted(train_files)]
        file_manifest["val"][class_name] = [str(path) for path in sorted(val_files)]
        file_manifest["test"][class_name] = [str(path) for path in sorted(test_files)]

        for split_name, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
            for path in split_files:
                info = read_wave_info(path)
                total_samples = infer_resampled_frame_count(
                    frame_count=info.frame_count,
                    source_rate_hz=info.sample_rate_hz,
                    target_rate_hz=sample_rate_hz,
                )
                starts = enumerate_segment_starts(
                    total_samples=total_samples,
                    window_samples=window_samples,
                    hop_samples=hop_samples,
                    pad_short=pad_short,
                )
                for start_sample in starts:
                    split_records[split_name].append(
                        SegmentRecord(
                            path=str(path),
                            label_index=class_index,
                            split=split_name,
                            start_sample=start_sample,
                            end_sample=start_sample + window_samples,
                        )
                    )

    if not split_records["train"]:
        raise RuntimeError("Training split is empty after indexing")
    if val_ratio > 0 and not split_records["val"]:
        raise RuntimeError("Validation split is empty after indexing")
    if test_ratio > 0 and not split_records["test"]:
        raise RuntimeError("Test split is empty after indexing")

    return SplitBundle(
        train=split_records["train"],
        val=split_records["val"],
        test=split_records["test"],
        file_manifest=file_manifest,
    )


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """把 Hz 频率映射到 mel 频率。"""

    return 2595.0 * torch.log10(1.0 + (hz / 700.0))


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """把 mel 频率映射回 Hz。"""

    return 700.0 * (torch.pow(torch.tensor(10.0, dtype=torch.float32), mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> torch.Tensor:
    """构建与部署合同一致的 mel 三角滤波器组。"""

    n_freqs = (n_fft // 2) + 1
    fft_freqs = torch.linspace(0.0, float(sample_rate_hz) / 2.0, n_freqs, dtype=torch.float32)
    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(float(fmin_hz), dtype=torch.float32)),
        hz_to_mel(torch.tensor(float(fmax_hz), dtype=torch.float32)),
        n_mels + 2,
        dtype=torch.float32,
    )
    hz_points = mel_to_hz(mel_points)
    filterbank = torch.zeros((n_mels, n_freqs), dtype=torch.float32)

    for mel_index in range(n_mels):
        left_hz = hz_points[mel_index]
        center_hz = hz_points[mel_index + 1]
        right_hz = hz_points[mel_index + 2]

        left_slope = (fft_freqs - left_hz) / max(float(center_hz - left_hz), 1e-6)
        right_slope = (right_hz - fft_freqs) / max(float(right_hz - center_hz), 1e-6)
        filterbank[mel_index] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return filterbank


class LogMelFeatureExtractor:
    """把 1 秒单通道波形转换成单样本 log-mel 特征图。"""

    def __init__(self, config: dict[str, Any]):
        audio_cfg = config["audio"]
        feature_cfg = config["feature"]
        self.sample_rate_hz = int(audio_cfg["sample_rate"])
        self.window_samples = int(round(self.sample_rate_hz * float(audio_cfg["window_sec"])))
        self.n_fft = int(feature_cfg["n_fft"])
        self.win_length = int(feature_cfg["win_length"])
        self.hop_length = int(feature_cfg["hop_length"])
        self.n_mels = int(feature_cfg["n_mels"])
        self.fmin_hz = float(feature_cfg["fmin"])
        self.fmax_hz = float(feature_cfg["fmax"])
        self.power = float(feature_cfg.get("power", 2.0))
        self.center = bool(feature_cfg.get("center", False))
        self.top_db = float(feature_cfg.get("top_db", 80.0))
        self.normalize = bool(feature_cfg.get("normalize", True))
        self.window = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.mel_filterbank = build_mel_filterbank(
            sample_rate_hz=self.sample_rate_hz,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin_hz=self.fmin_hz,
            fmax_hz=self.fmax_hz,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """执行 STFT -> mel -> dB -> normalize。"""

        waveform = waveform.to(dtype=torch.float32)
        if waveform.numel() != self.window_samples:
            raise ValueError(
                f"Feature extractor expects exactly {self.window_samples} samples, got {waveform.numel()}"
            )

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        magnitude = stft.abs()
        if self.power == 1.0:
            power_spec = magnitude
        else:
            power_spec = torch.pow(magnitude, self.power)

        mel_spec = torch.matmul(self.mel_filterbank, power_spec)
        mel_spec = torch.clamp(mel_spec, min=1e-10)
        db_spec = 10.0 * torch.log10(mel_spec)
        if self.top_db > 0:
            db_spec = torch.clamp(db_spec, min=float(db_spec.max()) - self.top_db)

        if self.normalize:
            mean = db_spec.mean()
            std = db_spec.std(unbiased=False)
            db_spec = (db_spec - mean) / max(float(std), 1e-6)

        return db_spec.unsqueeze(0)


class AudioWindowDataset(Dataset):
    """按切窗索引懒加载音频并生成模型输入特征。"""

    def __init__(
        self,
        records: list[SegmentRecord],
        feature_extractor: LogMelFeatureExtractor,
        augment_cfg: dict[str, Any] | None = None,
        *,
        training: bool = False,
    ):
        self.records = records
        self.feature_extractor = feature_extractor
        self.augment_cfg = augment_cfg or {}
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取单条样本，并按固定窗口规则输出 log-mel 特征。"""

        record = self.records[index]
        waveform, sample_rate_hz = load_wav_mono(Path(record.path))
        waveform = resample_waveform(
            waveform=waveform,
            source_rate_hz=sample_rate_hz,
            target_rate_hz=self.feature_extractor.sample_rate_hz,
        )

        window = waveform[record.start_sample:record.end_sample]
        if window.numel() < self.feature_extractor.window_samples:
            pad_length = self.feature_extractor.window_samples - window.numel()
            window = F.pad(window, (0, pad_length))
        elif window.numel() > self.feature_extractor.window_samples:
            window = window[:self.feature_extractor.window_samples]

        feature = self.feature_extractor(window)
        return feature, torch.tensor(record.label_index, dtype=torch.long)


def infer_feature_shape(config: dict[str, Any]) -> dict[str, int]:
    """提供当前输入配置下的固定特征尺寸。"""

    feature_extractor = LogMelFeatureExtractor(config)
    return {
        "window_samples": feature_extractor.window_samples,
        "n_mels": feature_extractor.n_mels,
        "frame_count": infer_feature_frame_count(
            sample_rate_hz=feature_extractor.sample_rate_hz,
            window_sec=float(config["audio"]["window_sec"]),
            n_fft=feature_extractor.n_fft,
            win_length=feature_extractor.win_length,
            hop_length=feature_extractor.hop_length,
            center=feature_extractor.center,
        ),
    }
