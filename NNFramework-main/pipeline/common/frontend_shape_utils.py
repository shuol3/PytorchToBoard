from __future__ import annotations


def infer_stft_analysis_frame_samples(
    *,
    n_fft: int | None,
    win_length: int | None,
) -> int:
    if n_fft is not None:
        return int(n_fft)
    if win_length is not None:
        return int(win_length)
    raise ValueError("At least one of n_fft or win_length must be provided")


def infer_feature_frame_count(
    *,
    sample_rate_hz: int,
    window_sec: float,
    hop_length: int,
    center: bool,
    n_fft: int | None = None,
    win_length: int | None = None,
) -> int:
    analysis_frame_samples = infer_stft_analysis_frame_samples(
        n_fft=n_fft,
        win_length=win_length,
    )
    window_samples = int(round(float(sample_rate_hz) * float(window_sec)))
    if bool(center):
        return 1 + (window_samples // int(hop_length))
    effective = max(window_samples - analysis_frame_samples, 0)
    return 1 + (effective // int(hop_length))


def infer_feature_input_shape_nchw(
    *,
    sample_rate_hz: int,
    window_sec: float,
    hop_length: int,
    n_mels: int,
    center: bool,
    n_fft: int | None = None,
    win_length: int | None = None,
) -> list[int]:
    return [
        1,
        1,
        int(n_mels),
        infer_feature_frame_count(
            sample_rate_hz=sample_rate_hz,
            window_sec=window_sec,
            hop_length=hop_length,
            center=center,
            n_fft=n_fft,
            win_length=win_length,
        ),
    ]
