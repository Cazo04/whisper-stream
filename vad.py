import torch
import numpy as np


class VADEngine:
    def __init__(
        self,
        threshold: float = 0.35,
        sampling_rate: int = 16000,
    ):
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self.threshold = threshold
        self.sampling_rate = sampling_rate

    def detect_speech(self, audio_float32: np.ndarray) -> bool:
        """Fast speech detection using direct model forward pass.

        Silero VAD model() requires exactly 512 samples per call at 16kHz.
        We process the audio in 512-sample windows and return True if any
        window exceeds the speech threshold.
        """
        window_size = 512  # Required by Silero VAD at 16kHz
        n = len(audio_float32)
        if n < window_size:
            return False

        audio_tensor = torch.from_numpy(audio_float32)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()

        # Process last few windows (most recent audio is most relevant)
        # Check up to 4 windows from the end for efficiency
        max_windows = 4
        start = max(0, n - window_size * max_windows)
        offset = start

        while offset + window_size <= n:
            window = audio_tensor[offset:offset + window_size]
            prob = self.model(window, self.sampling_rate).item()
            if prob >= self.threshold:
                return True
            offset += window_size

        return False

    def reset_state(self):
        """Reset VAD model hidden state between utterances."""
        self.model.reset_states()
