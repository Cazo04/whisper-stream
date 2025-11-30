import torch
import numpy as np


class VADEngine:
    def __init__(
        self,
        threshold: float = 0.35,
        sampling_rate: int = 16000,
        min_speech_ms: int = 80,
        min_silence_ms: int = 150,
        speech_pad_ms: int = 50,
    ):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, _, _, _) = utils

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

    def detect_speech(self, audio_float32: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_float32)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()

        timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        return len(timestamps) > 0
