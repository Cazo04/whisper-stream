# translator.py — HY-MT Translation Engine for Whisper Stream
# Tencent HY-MT1.5-7B, loaded once into VRAM alongside Whisper

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# LANGUAGE MAP (33+ ngôn ngữ)
# ─────────────────────────────────────────────
LANG_MAP = {
    "zh": "Chinese", "en": "English", "fr": "French",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese",
    "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
    "ko": "Korean", "th": "Thai", "it": "Italian",
    "de": "German", "vi": "Vietnamese", "ms": "Malay",
    "id": "Indonesian", "tl": "Filipino", "hi": "Hindi",
    "zh-Hant": "Traditional Chinese", "pl": "Polish",
    "cs": "Czech", "nl": "Dutch", "km": "Khmer",
    "my": "Burmese", "fa": "Persian", "gu": "Gujarati",
    "ur": "Urdu", "te": "Telugu", "mr": "Marathi",
    "he": "Hebrew", "bn": "Bengali", "ta": "Tamil",
    "uk": "Ukrainian", "bo": "Tibetan", "kk": "Kazakh",
    "mn": "Mongolian", "ug": "Uyghur", "yue": "Cantonese",
}


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))


class TranslatorEngine:
    """
    Singleton-style translation engine using Tencent HY-MT1.5-7B.
    Loaded once into VRAM, reused across all requests.
    """

    def __init__(
        self,
        model_name: str = "tencent/HY-MT1.5-7B",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        print(f"[HY-MT] Loading model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        print("[HY-MT] Model ready.")

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.6,
        top_k: int = 10,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Internal: generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template may return a BatchEncoding or a plain tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

        new_tokens = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return result

    def translate(
        self,
        text: str,
        target_lang: str = "vi",
        context: str | None = None,
    ) -> str:
        """
        Translate text to target language.
        If context is provided, uses contextual translation for coherence.

        Args:
            text:        Text to translate.
            target_lang: Target language code (default "vi").
            context:     Previous sentences for contextual translation.
        Returns:
            Translated string.
        """
        if not text or not text.strip():
            return ""

        target_name = LANG_MAP.get(target_lang, target_lang)

        if context and context.strip():
            # Contextual translation — uses previous sentences as reference
            prompt = (
                f"{context}\n"
                f"参考上面的信息，把下面的文本翻译成{target_name}，"
                f"注意不需要翻译上文，也不要额外解释：\n{text}"
            )
        elif _has_chinese(text):
            # Chinese source → specialized ZH prompt
            prompt = (
                f"将以下文本翻译为{target_name}，"
                f"注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
            )
        else:
            # Generic source → English prompt
            prompt = (
                f"Translate the following segment into {target_name}, "
                f"without additional explanation.\n\n{text}"
            )

        return self._generate(prompt)
