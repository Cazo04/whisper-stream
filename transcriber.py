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

        self._language = None  # None = auto-detect

        self.default_kwargs = dict(
            language=None,
            task="transcribe",
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False,  # External VAD already handles speech detection
        )

    def set_language(self, lang: str | None):
        """Pin language to skip auto-detection (~15% faster per inference).
        Set to None or 'auto' to re-enable auto-detection.
        """
        if lang is None or lang == "auto":
            self._language = None
        else:
            self._language = lang
        self.default_kwargs["language"] = self._language
        print(f"Whisper language set to: {self._language or 'auto-detect'}")

    def get_language(self) -> str | None:
        return self._language

    def transcribe(self, audio: np.ndarray):
        segments, info = self.model.transcribe(audio, **self.default_kwargs)
        text = "".join(seg.text for seg in segments).strip()
        return text, info


class StreamingSession:
    """Buffers audio and transcribes only once at end-of-utterance (final only)."""

    def __init__(
        self,
        engine: WhisperEngine,
        sample_rate: int = 16000,
        max_buffer_sec: float = 15.0,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.max_buffer_len = int(max_buffer_sec * sample_rate)

        # Pre-allocate ring buffer for zero-copy audio accumulation
        self._buffer = np.zeros(self.max_buffer_len, dtype=np.float32)
        self._write_pos = 0

    def add_chunk_f32(self, chunk_f32: np.ndarray):
        """Append float32 audio directly (avoids double int16→float32 conversion)."""
        n = len(chunk_f32)
        if n == 0:
            return

        space_left = self.max_buffer_len - self._write_pos
        if n <= space_left:
            self._buffer[self._write_pos:self._write_pos + n] = chunk_f32
            self._write_pos += n
        else:
            # Buffer would overflow: shift left to make room
            keep = self.max_buffer_len - n
            if keep > 0 and self._write_pos > 0:
                self._buffer[:keep] = self._buffer[self._write_pos - keep:self._write_pos]
                self._buffer[keep:keep + n] = chunk_f32
                self._write_pos = keep + n
            else:
                # Chunk alone fills or exceeds buffer — keep only tail
                self._buffer[:self.max_buffer_len] = chunk_f32[-self.max_buffer_len:]
                self._write_pos = self.max_buffer_len

    @property
    def audio_buffer(self) -> np.ndarray:
        """Return the valid portion of the ring buffer (read-only view)."""
        return self._buffer[:self._write_pos]

    @property
    def duration_sec(self) -> float:
        return self._write_pos / self.sample_rate

    def transcribe_full(self):
        """Single transcription at end-of-utterance — the only Whisper call."""
        if self._write_pos == 0:
            return "", True

        try:
            audio = self._buffer[:self._write_pos]
            text, info = self.engine.transcribe(audio)
            if info is not None:
                print(
                    f"[FINAL] Detected language: {info.language} "
                    f"(probability: {info.language_probability:.2f}), "
                    f"audio: {self._write_pos / self.sample_rate:.1f}s"
                )
            return text, True
        except ValueError as e:
            if "empty sequence" in str(e):
                return "", True
            raise
        except Exception as e:
            print(f"Unexpected error during transcription: {e}")
            return "", True

    def reset(self):
        self._write_pos = 0
