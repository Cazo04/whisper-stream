# translator.py
# pip install transformers==4.56.0 torch langdetect

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# LANGUAGE MAP (33 ngôn ngữ hỗ trợ)
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

# ─────────────────────────────────────────────
# MODEL LOADER – chỉ tải 1 lần, tái sử dụng
# ─────────────────────────────────────────────
_model = None
_tokenizer = None

def load_model(
    model_name: str = "tencent/HY-MT1.5-7B",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> tuple:
    """
    Tải model và tokenizer vào bộ nhớ.
    Gọi một lần duy nhất; các lần sau trả về instance đã tải.
    
    Args:
        model_name: HuggingFace model ID hoặc đường dẫn local.
                    Dùng "tencent/HY-MT1.5-1.8B" nếu VRAM < 16GB.
        dtype:      torch.bfloat16 (khuyên dùng) hoặc torch.float16.
        device_map: "auto" tự phân bổ GPU/CPU.
    Returns:
        (model, tokenizer)
    """
    global _model, _tokenizer
    if _model is None:
        print(f"[HY-MT] Đang tải model: {model_name} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        _model.eval()
        print("[HY-MT] Model sẵn sàng.")
    return _model, _tokenizer


# ─────────────────────────────────────────────
# CORE: SINH VĂN BẢN
# ─────────────────────────────────────────────
def _generate(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.6,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
) -> str:
    """Nội bộ: gọi model sinh văn bản từ prompt."""
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    # Chỉ lấy phần được sinh ra (bỏ prompt)
    new_tokens = outputs[0][inputs.shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return result


def _has_chinese(text: str) -> bool:
    """Kiểm tra văn bản có chứa chữ Hán không."""
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))


# ─────────────────────────────────────────────
# HÀM 1: DỊCH CƠ BẢN (tự động nhận mọi nguồn)
# ─────────────────────────────────────────────
def translate(
    text: str,
    target_lang: str = "vi",
    **gen_kwargs,
) -> str:
    """
    Dịch văn bản bất kỳ → ngôn ngữ đích.
    Tự động chọn prompt template phù hợp (ZH↔XX hay XX↔XX).

    Args:
        text:        Văn bản cần dịch.
        target_lang: Mã ngôn ngữ đích (mặc định "vi" = tiếng Việt).
        **gen_kwargs: Tham số sinh văn bản tùy chọn.
    Returns:
        Chuỗi đã dịch.

    Example:
        >>> translate("Hello world!")
        'Xin chào thế giới!'
        >>> translate("こんにちは、世界！")
        'Xin chào thế giới!'
    """
    target_name = LANG_MAP.get(target_lang, target_lang)

    if _has_chinese(text):
        # Template dành riêng cho ZH ↔ XX (theo tài liệu chính thức)
        prompt = (
            f"将以下文本翻译为{target_name}，"
            f"注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        )
    else:
        # Template cho mọi cặp ngôn ngữ còn lại
        prompt = (
            f"Translate the following segment into {target_name}, "
            f"without additional explanation.\n\n{text}"
        )

    return _generate(prompt, **gen_kwargs)


# ─────────────────────────────────────────────
# HÀM 2: DỊCH VĂN BẢN PHA NHIỀU NGÔN NGỮ
# ─────────────────────────────────────────────
def translate_mixed(
    text: str,
    target_lang: str = "vi",
    hint: str = "",
    **gen_kwargs,
) -> str:
    """
    Dịch văn bản chứa nhiều ngôn ngữ trộn lẫn trong 1 câu/đoạn.
    HY-MT1.5-7B được tối ưu đặc biệt cho tình huống này.

    Args:
        text:        Đoạn văn pha trộn ngôn ngữ.
        target_lang: Ngôn ngữ đích (mặc định "vi").
        hint:        Gợi ý thêm về ngữ cảnh (tuỳ chọn).
        **gen_kwargs: Tham số sinh văn bản.
    Returns:
        Bản dịch tiếng Việt tự nhiên, đúng nghĩa.

    Example:
        >>> translate_mixed("I want to eat phở because it's très délicieux!")
        'Tôi muốn ăn phở vì nó rất ngon!'

        >>> translate_mixed("我今天went to the 市場 to buy some đồ ăn")
        'Hôm nay tôi đi chợ để mua thức ăn.'
    """
    target_name = LANG_MAP.get(target_lang, target_lang)
    hint_line = f"\nAdditional context: {hint}" if hint else ""

    prompt = (
        f"The following text contains a mix of multiple languages.{hint_line}\n"
        f"Translate the entire text into {target_name} naturally and accurately, "
        f"preserving the original meaning. Output only the translation.\n\n{text}"
    )
    return _generate(prompt, **gen_kwargs)


# ─────────────────────────────────────────────
# HÀM 3: DỊCH CÓ NGỮ CẢNH (contextual)
# ─────────────────────────────────────────────
def translate_with_context(
    text: str,
    context: str,
    target_lang: str = "vi",
    **gen_kwargs,
) -> str:
    """
    Dịch câu/đoạn dựa trên ngữ cảnh (ví dụ: đoạn trước đó).
    Giúp dịch nhất quán, đúng với mạch văn.

    Args:
        text:        Đoạn văn cần dịch.
        context:     Các câu/đoạn phía trước làm ngữ cảnh tham chiếu.
        target_lang: Ngôn ngữ đích.
        **gen_kwargs: Tham số sinh.
    Returns:
        Bản dịch có tính nhất quán ngữ cảnh cao.

    Example:
        >>> ctx = "The protagonist, Minh, is a young engineer in Hanoi."
        >>> translate_with_context("He loves his job deeply.", context=ctx)
        'Anh ấy yêu công việc của mình sâu sắc.'
    """
    target_name = LANG_MAP.get(target_lang, target_lang)

    # Prompt contextual theo tài liệu chính thức
    prompt = (
        f"{context}\n"
        f"参考上面的信息，把下面的文本翻译成{target_name}，"
        f"注意不需要翻译上文，也不要额外解释：\n{text}"
    )
    return _generate(prompt, **gen_kwargs)


# ─────────────────────────────────────────────
# HÀM 4: DỊCH CÓ THUẬT NGỮ BẮT BUỘC
# ─────────────────────────────────────────────
def translate_with_terminology(
    text: str,
    terms: dict[str, str],
    target_lang: str = "vi",
    **gen_kwargs,
) -> str:
    """
    Dịch với danh sách thuật ngữ bắt buộc (terminology intervention).
    Đảm bảo tên riêng, thuật ngữ kỹ thuật được dịch nhất quán.

    Args:
        text:        Văn bản cần dịch.
        terms:       Dict {từ_gốc: từ_dịch} ép buộc dùng.
        target_lang: Ngôn ngữ đích.
        **gen_kwargs: Tham số sinh.
    Returns:
        Bản dịch có sử dụng đúng thuật ngữ chỉ định.

    Example:
        >>> translate_with_terminology(
        ...     "The transformer model is state-of-the-art.",
        ...     terms={"transformer": "biến thể"},
        ... )
        'Mô hình biến thể là tiên tiến nhất.'
    """
    target_name = LANG_MAP.get(target_lang, target_lang)
    term_lines = "\n".join(
        f"{src} 翻译成 {tgt}" for src, tgt in terms.items()
    )

    prompt = (
        f"参考下面的翻译：\n{term_lines}\n\n"
        f"将以下文本翻译为{target_name}，"
        f"注意只需要输出翻译后的结果，不要额外解释：\n{text}"
    )
    return _generate(prompt, **gen_kwargs)


# ─────────────────────────────────────────────
# HÀM 5: DỊCH HÀNG LOẠT (batch)
# ─────────────────────────────────────────────
def translate_batch(
    texts: list[str],
    target_lang: str = "vi",
    mode: str = "auto",
    **gen_kwargs,
) -> list[str]:
    """
    Dịch nhiều đoạn văn cùng lúc.

    Args:
        texts:       Danh sách chuỗi cần dịch.
        target_lang: Ngôn ngữ đích.
        mode:        "auto" (thường), "mixed" (pha ngôn ngữ).
        **gen_kwargs: Tham số sinh.
    Returns:
        List bản dịch tương ứng.

    Example:
        >>> results = translate_batch(["Hello!", "Bonjour!", "こんにちは！"])
        >>> print(results)
        ['Xin chào!', 'Xin chào!', 'Xin chào!']
    """
    fn = translate_mixed if mode == "mixed" else translate
    return [fn(t, target_lang=target_lang, **gen_kwargs) for t in texts]
