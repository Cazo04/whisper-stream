import numpy as np
from faster_whisper import WhisperModel


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            print("Failed to load on CUDA, falling back to CPU int8.")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self.default_kwargs = dict(
            language=None,
            task="transcribe",
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

    def transcribe(self, audio: np.ndarray):
        segments, info = self.model.transcribe(audio, **self.default_kwargs)
        text = "".join(seg.text for seg in segments).strip()
        return text, info


class StreamingSession:
    def __init__(
        self,
        engine: WhisperEngine,
        sample_rate: int = 16000,
        max_buffer_sec: float = 10.0,
        min_inference_sec: float = 1.0,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.max_buffer_len = int(max_buffer_sec * sample_rate)
        self.min_inference_len = int(min_inference_sec * sample_rate)

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_inferred_len = 0

    def add_chunk(self, chunk_bytes: bytes):
        audio_array = (
            np.frombuffer(chunk_bytes, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

        self.audio_buffer = np.concatenate((self.audio_buffer, audio_array))

        if len(self.audio_buffer) > self.max_buffer_len:
            dropped = len(self.audio_buffer) - self.max_buffer_len
            self.audio_buffer = self.audio_buffer[-self.max_buffer_len:]
            self.last_inferred_len = max(0, self.last_inferred_len - dropped)

    def transcribe_partial(self):
        n_samples = len(self.audio_buffer)
        if n_samples < self.min_inference_len:
            return "", False

        new_samples = n_samples - self.last_inferred_len
        if new_samples < 0:
            new_samples = n_samples
            self.last_inferred_len = 0

        if new_samples < self.min_inference_len // 2:
            return "", False

        try:
            text, info = self.engine.transcribe(self.audio_buffer)
            self.last_inferred_len = n_samples

            if info is not None:
                print(
                    f"Detected language: {info.language} "
                    f"(probability: {info.language_probability:.2f})"
                )

            return text, False  # partial
        except ValueError as e:
            if "empty sequence" in str(e):
                return "", False
            raise
        except Exception as e:
            print(f"Unexpected error during transcription: {e}")
            return "", False

    def transcribe_full(self):
        n_samples = len(self.audio_buffer)
        if n_samples == 0:
            return "", True

        try:
            text, info = self.engine.transcribe(self.audio_buffer)
            if info is not None:
                print(
                    f"[FINAL] Detected language: {info.language} "
                    f"(probability: {info.language_probability:.2f})"
                )
            return text, True
        except ValueError as e:
            if "empty sequence" in str(e):
                return "", True
            raise
        except Exception as e:
            print(f"Unexpected error during FINAL transcription: {e}")
            return "", True

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_inferred_len = 0
