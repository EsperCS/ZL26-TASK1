"""
src/asr/sherpa_xlm_pipeline.py

Pipeline ASR 4 tầng thế hệ mới (Next-Gen 4-Tier Vietnamese ASR):
1. Tầng 1: Audio Normalizer & Silero VAD (Sherpa-ONNX):
   - Đọc WAV 16kHz Mono.
   - Chuẩn hóa Peak Amplitude về [-0.95, 0.95].
   - Tách các đoạn giọng nói và cắt bỏ khoảng lặng bằng Silero VAD (min_silence_duration=0.4s).
2. Tầng 2: Transducer ASR (Sherpa-ONNX):
   - Hỗ trợ Gipformer-68M hoặc ZipFormer-150M.
   - Decoding method: modified_beam_search, max_active_paths=4.
   - Ghép các segment và chuyển thành raw_acoustic_text (.lower().strip()).
3. Tầng 3: Punctuation & Capitalization Restoration (XLM-RoBERTa Capu):
   - Chạy dragonSwing/xlm-roberta-capu để khôi phục viết hoa tên riêng và dấu câu (. , : ?).
4. Tầng 4: Vietnamese ITN (Inverse Text Normalization):
   - Chuẩn hóa số tiếng Việt sang định dạng số thông dụng (thời gian, thời lượng, tiền tệ, số đếm)
   - Phục vụ mapping tham số chính xác cho ToolEnv Lifemate.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import sherpa_onnx
import soundfile as sf

# ============================================================================
# TẦNG 4: VIETNAMESE INVERSE TEXT NORMALIZATION (ITN)
# ============================================================================

class VietnameseITN:
    """
    Quy tắc chuẩn hóa ngược (Inverse Text Normalization) cho tiếng Việt:
    Chuyển đổi số nói (spoken numbers) thành số viết (digits) kết hợp cùng đơn vị.
    Ví dụ:
      - "một tiếng" -> "1 tiếng", "hai tiếng" -> "2 tiếng", "nửa tiếng" -> "nửa tiếng"
      - "năm mươi nghìn" -> "50 nghìn", "năm mươi ngàn" -> "50 ngàn"
      - "hai trăm nghìn" -> "200 nghìn", "hai triệu" -> "2 triệu"
      - "chín giờ" -> "9 giờ", "5 rưỡi" -> "5 rưỡi", "chín giờ ba mươi" -> "9 giờ 30"
      - "tầng hai" -> "tầng 2", "ngày mười lăm" -> "ngày 15"
    """

    _DIGITS: ClassVar[dict[str, int]] = {
        "không": 0,
        "một": 1,
        "mốt": 1,
        "hai": 2,
        "ba": 3,
        "bốn": 4,
        "tư": 4,
        "năm": 5,
        "lăm": 5,
        "sáu": 6,
        "bảy": 7,
        "bẩy": 7,
        "tám": 8,
        "chín": 9,
    }

    _TEENS: ClassVar[dict[str, int]] = {
        "mười": 10,
        "mười một": 11,
        "mười hai": 12,
        "mười ba": 13,
        "mười bốn": 14,
        "mười lăm": 15,
        "mười sáu": 16,
        "mười bảy": 17,
        "mười bẩy": 17,
        "mười tám": 18,
        "mười chín": 19,
    }

    _TENS: ClassVar[dict[str, int]] = {
        "hai mươi": 20,
        "ba mươi": 30,
        "bốn mươi": 40,
        "năm mươi": 50,
        "sáu mươi": 60,
        "bảy mươi": 70,
        "bẩy mươi": 70,
        "tám mươi": 80,
        "chín mươi": 90,
        # Dạng tắt thông dụng trong khẩu ngữ
        "hăm": 20,
        "hai mốt": 21,
        "hai lăm": 25,
        "ba lăm": 35,
        "bốn lăm": 45,
        "năm lăm": 55,
    }

    @classmethod
    def parse_spoken_number(cls, phrase: str) -> int | None:
        """Chuyển một cụm từ tiếng Việt diễn tả số (0-999) thành số nguyên."""
        phrase = phrase.strip().lower()
        if not phrase:
            return None

        # 1. Trực tiếp là số
        if phrase.isdigit():
            return int(phrase)

        # 2. Chữ số đơn
        if phrase in cls._DIGITS:
            return cls._DIGITS[phrase]

        # 3. Hàng mười (10-19)
        if phrase in cls._TEENS:
            return cls._TEENS[phrase]

        # 4. Hàng chục tròn (20, 30, ...)
        if phrase in cls._TENS:
            return cls._TENS[phrase]

        # 5. Ghép "hàng chục + đơn vị" (e.g. "hai mươi ba", "năm mươi mốt")
        for tens_name, tens_val in cls._TENS.items():
            if phrase.startswith(tens_name + " "):
                unit_part = phrase[len(tens_name) :].strip()
                if unit_part in cls._DIGITS:
                    return tens_val + cls._DIGITS[unit_part]

        # 6. Ghép hàng trăm (e.g. "một trăm", "một trăm hai mươi ba")
        match_hundred = re.match(r"^(một|hai|ba|bốn|năm|sáu|bảy|bẩy|tám|chín)\s+trăm(?:\s+(.*))?$", phrase)
        if match_hundred:
            hundred_digit = cls._DIGITS[match_hundred.group(1)]
            rem = match_hundred.group(2)
            if not rem or not rem.strip():
                return hundred_digit * 100
            rem = rem.strip()
            # Dạng "lẻ năm", "linh năm"
            match_le = re.match(r"^(?:lẻ|linh)\s+(\w+)$", rem)
            if match_le and match_le.group(1) in cls._DIGITS:
                return hundred_digit * 100 + cls._DIGITS[match_le.group(1)]
            # Dạng phần chục
            rem_val = cls.parse_spoken_number(rem)
            if rem_val is not None:
                return hundred_digit * 100 + rem_val

        return None

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Chuẩn hóa văn bản sau ASR / Capu bằng quy tắc ITN.
        """
        if not text:
            return ""

        result = text

        # 1. Chuẩn hóa đơn vị đo lường, thời gian, tiền tệ sau số nói:
        # vd: "một tiếng", "hai mươi ba triệu", "năm mươi nghìn", "chín giờ", "tầng hai"
        units_pattern = r"(tiếng|giờ|phút|giây|nghìn|ngàn|triệu|tỷ|ngày|tháng|năm|tuần|lần|độ|phần trăm|km|m)"

        # Regex tìm các cụm số nói đứng trước đơn vị (tối đa 4 từ tiếng Việt)
        # Bỏ qua từ "nửa" (giữ nguyên "nửa tiếng", "nửa ngày")
        def replace_number_before_unit(m: re.Match[str]) -> str:
            prefix = m.group(1)  # khoảng trắng hoặc ranh giới từ
            num_words = m.group(2).strip()
            unit = m.group(3)

            # Ngoại lệ: "thứ hai" (Monday), "bác Ba", "cô Năm"
            if num_words in {"nửa", "một nửa"}:
                return m.group(0)

            parsed = cls.parse_spoken_number(num_words)
            if parsed is not None:
                return f"{prefix}{parsed} {unit}"
            return m.group(0)

        # Hỗ trợ cụm 1 đến 4 từ số trước đơn vị
        pattern_unit = rf"(^|[\s,;.(])((?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|trăm|mươi|linh|lẻ|\d+)(?:\s+(?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|trăm|mươi|linh|lẻ|\d+)){{0,3}})\s+{units_pattern}\b"
        result = re.sub(pattern_unit, replace_number_before_unit, result, flags=re.IGNORECASE)

        # 2. Chuẩn hóa "tầng + số" -> "tầng 2"
        def replace_floor(m: re.Match[str]) -> str:
            word = m.group(2).strip().lower()
            parsed = cls.parse_spoken_number(word)
            if parsed is not None:
                return f"{m.group(1)} {parsed}"
            return m.group(0)

        result = re.sub(
            r"\b(tầng|lầu|căn|phòng|số)\s+((?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|\d+))\b",
            replace_floor,
            result,
            flags=re.IGNORECASE,
        )

        # 3. Chuẩn hóa giờ rưỡi: "năm rưỡi" -> "5 rưỡi", "chín rưỡi" -> "9 rưỡi"
        def replace_half_hour(m: re.Match[str]) -> str:
            num_word = m.group(1).lower()
            parsed = cls.parse_spoken_number(num_word)
            if parsed is not None:
                return f"{parsed} rưỡi"
            return m.group(0)

        result = re.sub(
            r"\b(một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín|mười|mười một|mười hai)\s+rưỡi\b",
            replace_half_hour,
            result,
            flags=re.IGNORECASE,
        )

        # 4. Chuẩn hóa dời/hoãn: "dời ... muộn 1 tiếng", "muộn hai tiếng"
        # 5. Dọn dẹp khoảng trắng kép
        result = re.sub(r"\s+", " ", result).strip()

        return result


# ============================================================================
# TẦNG 1-3: PIPELINE TỔNG HỢP SHERPA-ONNX + XLM-ROBERTA CAPU
# ============================================================================

@dataclass(slots=True, kw_only=True)
class ASRTranscriptionResult:
    raw_acoustic_text: str
    capu_text: str
    final_text: str
    duration_sec: float
    inference_time_sec: float
    segments_count: int
    sample_rate: int = 16000


class SherpaXLMPipeline:
    """
    Pipeline ASR 4 tầng:
    Silero VAD + Sherpa-ONNX Transducer + XLM-RoBERTa Capu + Vietnamese ITN.
    """

    def __init__(
        self,
        *,
        model_name_or_dir: str | Path,
        vad_model_path: str | Path = "models/asr/vad/silero_vad.onnx",
        capu_model_dir: str | Path | None = "models/asr/xlm-roberta-capu",
        capu_model: Any = None,
        device: str = "cpu",
        num_threads: int = 4,
        enable_capu: bool = True,
        enable_itn: bool = True,
        vad_min_silence: float = 0.4,
        vad_min_speech: float = 0.25,
        vad_threshold: float = 0.5,
        hotwords_file: str | Path | None = None,
        hotwords_score: float = 1.5,
        provider: str = "cpu",
    ) -> None:
        self.device = device
        self.enable_capu = enable_capu
        self.enable_itn = enable_itn
        self.itn = VietnameseITN()
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        self.provider = provider

        model_path = Path(model_name_or_dir)
        vad_path = Path(vad_model_path)

        # 1. Khởi tạo Silero VAD (Sherpa-ONNX, tự động tải nếu thiếu)
        if not vad_path.exists() or vad_path.stat().st_size < 500_000:
            self._ensure_vad_file(vad_path)

        silero_config = sherpa_onnx.SileroVadModelConfig(
            model=str(vad_path),
            threshold=vad_threshold,
            min_silence_duration=vad_min_silence,
            min_speech_duration=vad_min_speech,
            window_size=512,
        )
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=silero_config,
            sample_rate=16000,
            num_threads=1,
            provider="cpu",
        )
        self.vad_config = vad_config

        # 2. Khởi tạo Transducer Offline Recognizer
        self.recognizer = self._init_transducer_recognizer(
            model_path,
            num_threads=num_threads,
            hotwords_file=hotwords_file,
            hotwords_score=hotwords_score,
            provider=provider,
        )

        # 3. Khởi tạo hoặc tái sử dụng XLM-RoBERTa Capu
        self.capu_model = None
        if enable_capu:
            if capu_model is not None:
                self.capu_model = capu_model
            elif capu_model_dir:
                capu_path = Path(capu_model_dir)
                has_weights = (
                    (capu_path / "pytorch_model.bin").exists()
                    or (capu_path / "model.safetensors").exists()
                    or any(capu_path.glob("*.safetensors"))
                )
                if capu_path.exists() and has_weights:
                    self.capu_model = self._init_capu_model(capu_path, device=device)
                else:
                    print(f"[CẢNH BÁO] Chưa tìm thấy đầy đủ weights Capu tại {capu_path}. Tạm tắt tầng Capu.")

    @staticmethod
    def _ensure_vad_file(vad_path: Path) -> Path:
        """Tự động kiểm tra và tải silero_vad.onnx nếu chưa tồn tại cục bộ."""
        if vad_path.exists() and vad_path.stat().st_size > 500_000:
            return vad_path

        vad_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        print(f"[AUTO-DOWNLOAD] Đang tải silero_vad.onnx từ {url} về '{vad_path.resolve()}'...")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, str(vad_path))
            print(f"[THÀNH CÔNG] Đã tải silero_vad.onnx ({vad_path.stat().st_size} bytes)")
        except Exception as e:
            raise FileNotFoundError(
                f"Không tìm thấy file VAD tại {vad_path} và không thể tự động tải từ {url}: {e}"
            ) from e
        return vad_path

    @staticmethod
    def _ensure_transducer_model(model_dir: Path) -> None:
        """Tự động kiểm tra và tải mô hình ZipFormer hoặc Gipformer từ Hugging Face nếu thiếu."""
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_files = [f for f in model_dir.iterdir() if f.suffix == ".onnx"] if model_dir.exists() else []
        if onnx_files:
            cfg = model_dir / "config.json"
            tok = model_dir / "tokens.txt"
            if cfg.exists() and not tok.exists():
                import shutil

                shutil.copy(cfg, tok)
            return

        model_name = str(model_dir).lower()
        if "gipformer" in model_name:
            repo_id = "g-group-ai-lab/gipformer1.5-68M-rnnt"
        else:
            repo_id = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"

        print(f"[AUTO-DOWNLOAD] Đang tải mô hình Transducer '{repo_id}' về '{model_dir.resolve()}'...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                allow_patterns=["*.onnx", "config.json", "tokens.txt", "bpe.model", "README.md"],
                resume_download=True,
            )
            cfg = model_dir / "config.json"
            tok = model_dir / "tokens.txt"
            if cfg.exists() and not tok.exists():
                import shutil

                shutil.copy(cfg, tok)
            print(f"[THÀNH CÔNG] Đã tải hoàn tất mô hình Transducer '{repo_id}'")
        except Exception as e:  # noqa: BLE001
            print(f"[CẢNH BÁO] Không thể tự động tải mô hình Transducer {repo_id}: {e}")

    def _init_transducer_recognizer(
        self,
        model_dir: Path,
        num_threads: int,
        hotwords_file: str | Path | None = None,
        hotwords_score: float = 1.5,
        provider: str = "cpu",
    ) -> sherpa_onnx.OfflineRecognizer:
        """Tìm file ONNX và khởi tạo sherpa_onnx.OfflineRecognizer (hỗ trợ hotwords context biasing)."""
        self._ensure_transducer_model(model_dir)

        files = list(model_dir.iterdir())
        onnx_files = [f for f in files if f.suffix == ".onnx"]

        def pick_best(prefix: str) -> Path:
            candidates = [f for f in onnx_files if f.name.startswith(prefix)]
            if not candidates:
                raise FileNotFoundError(f"Không tìm thấy file {prefix}*.onnx trong {model_dir}")
            non_int8 = [f for f in candidates if ".int8." not in f.name]
            return non_int8[0] if non_int8 else candidates[0]

        encoder = pick_best("encoder")
        decoder = pick_best("decoder")
        joiner = pick_best("joiner")

        tokens = model_dir / "tokens.txt"
        if not tokens.exists():
            cfg = model_dir / "config.json"
            if cfg.exists():
                tokens = cfg
            else:
                raise FileNotFoundError(f"Không tìm thấy tokens.txt hoặc config.json trong {model_dir}")

        # Tự động xuất bpe.vocab nếu model có bpe.model nhưng chưa có file vocab dạng text
        bpe_model = model_dir / "bpe.model"
        bpe_vocab = model_dir / "bpe.vocab"
        if bpe_model.exists() and not bpe_vocab.exists():
            try:
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor()
                sp.load(str(bpe_model))
                with open(bpe_vocab, "w", encoding="utf-8") as f:
                    f.writelines(f"{sp.id_to_piece(i)} {sp.get_score(i)}\n" for i in range(sp.get_piece_size()))
            except Exception as e:  # noqa: BLE001
                print(f"[CẢNH BÁO] Không thể sinh bpe.vocab từ bpe.model: {e}")

        recognizer_kwargs: dict[str, Any] = {
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
            "tokens": str(tokens),
            "num_threads": num_threads,
            "sample_rate": 16000,
            "decoding_method": "modified_beam_search",
            "max_active_paths": 4,
            "provider": provider,
        }

        if hotwords_file:
            hw_path = Path(hotwords_file)
            if hw_path.exists():
                recognizer_kwargs["hotwords_file"] = str(hw_path)
                recognizer_kwargs["hotwords_score"] = float(hotwords_score)
                if bpe_vocab.exists():
                    recognizer_kwargs["bpe_vocab"] = str(bpe_vocab)
                    recognizer_kwargs["modeling_unit"] = "bpe"

        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**recognizer_kwargs)
        except Exception as e:
            if provider != "cpu":
                print(f"[CẢNH BÁO] Sherpa-ONNX không hỗ trợ provider='{provider}' ({e}). Tự động fallback về 'cpu'.")
                recognizer_kwargs["provider"] = "cpu"
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**recognizer_kwargs)
            else:
                raise
        return recognizer

    def _init_capu_model(self, capu_dir: Path, device: str) -> Any:
        """Nạp GecBERTModel từ repo dragonSwing/xlm-roberta-capu (kèm tự động vá tương thích transformers)."""
        abs_capu_dir = str(capu_dir.resolve())

        # Tự động vá tương thích mã nguồn cũ của dragonSwing với transformers 4.40+ / 5.x
        self._patch_capu_compatibility(capu_dir)

        if abs_capu_dir not in sys.path:
            sys.path.insert(0, abs_capu_dir)

        try:
            from gec_model import GecBERTModel

            capu = GecBERTModel(
                vocab_path=str(capu_dir / "vocabulary"),
                model_paths=[abs_capu_dir],
                device=device,
                split_chunk=True,
            )
            return capu
        except Exception as e:  # noqa: BLE001
            print(f"[CẢNH BÁO] Không thể nạp GecBERTModel ({e}). Tạm tắt tầng Capu.")
            return None

    @staticmethod
    def _patch_capu_compatibility(capu_dir: Path) -> None:
        """Tự động khắc phục tương thích mã nguồn cũ của dragonSwing/xlm-roberta-capu với transformers mới."""
        model_file = capu_dir / "modeling_seq2labels.py"
        if not model_file.exists():
            return
        try:
            content = model_file.read_text(encoding="utf-8")
            modified = False

            if "@dataclass" not in content:
                content = "from dataclasses import dataclass\n" + content
                content = content.replace(
                    "class Seq2LabelsOutput(ModelOutput):",
                    "@dataclass\nclass Seq2LabelsOutput(ModelOutput):",
                )
                modified = True

            if "mean_resizing=False" not in content and "self.bert.resize_token_embeddings(vocab_size + 1)" in content:
                content = content.replace(
                    "self.bert.resize_token_embeddings(vocab_size + 1)",
                    "self.bert.resize_token_embeddings(vocab_size + 1, mean_resizing=False)",
                )
                modified = True

            if "self.num_labels = getattr(config, \"vocab_size\", config.num_labels)" not in content:
                content = content.replace(
                    "self.num_labels = config.num_labels",
                    "self.num_labels = getattr(config, \"vocab_size\", config.num_labels)",
                )
                modified = True

            if modified:
                model_file.write_text(content, encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass

    def segment_speech_with_vad(self, samples: np.ndarray, sample_rate: int = 16000) -> list[np.ndarray]:
        """
        Dùng Silero VAD cắt bỏ khoảng lặng và trích xuất các speech segments.
        Nếu không phát hiện được tiếng nói nào, trả về toàn bộ mảng audio gốc.
        """
        # Peak amplitude normalization về [-0.95, 0.95]
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-4:
            samples = samples * (0.95 / peak)

        vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=60)
        window_size = 512
        float_samples = samples.astype(np.float32)

        # Cần feed theo từng chunk window_size để VAD kích hoạt bộ lọc đúng cách
        for i in range(0, len(float_samples), window_size):
            chunk = float_samples[i : i + window_size]
            vad.accept_waveform(chunk)
        vad.flush()

        segments: list[np.ndarray] = []
        while not vad.empty():
            seg = vad.front
            segments.append(np.array(seg.samples, dtype=np.float32))
            vad.pop()

        if not segments:
            # Fallback nếu VAD quá nghiêm ngặt
            segments = [float_samples]

        return segments

    def transcribe(
        self,
        audio_input: str | Path | np.ndarray,
        sample_rate: int = 16000,
    ) -> ASRTranscriptionResult:
        """
        Thực thi toàn bộ pipeline 4 tầng trên một file âm thanh hoặc mảng float.
        """
        t0 = time.perf_counter()

        # 1. Đọc audio (hỗ trợ .wav, .m4a, .mp3, .flac, .ogg qua ffmpeg fallback)
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")
            try:
                samples, sr = sf.read(str(audio_path), dtype="float32")
            except Exception:  # noqa: BLE001
                # Fallback qua ffmpeg để giải mã các định dạng m4a, mp3, ogg, aac...
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    tmp_wav = tmp_f.name
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(audio_path),
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            "-f",
                            "wav",
                            tmp_wav,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    samples, sr = sf.read(tmp_wav, dtype="float32")
                finally:
                    Path(tmp_wav).unlink(missing_ok=True)

            if samples.ndim > 1:
                samples = samples.mean(axis=1)  # Chuyển stereo -> mono
            if sr != 16000:
                # Nếu sample rate khác 16kHz, resample
                num_target = int(len(samples) * 16000 / sr)
                samples = np.interp(
                    np.linspace(0, len(samples), num_target, endpoint=False),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.float32)
                sr = 16000
        else:
            samples = np.asarray(audio_input, dtype=np.float32)
            sr = sample_rate

        duration_sec = float(len(samples)) / float(sr)

        # 2. Tầng 1: VAD Segmentation
        speech_segments = self.segment_speech_with_vad(samples, sample_rate=sr)

        # 3. Tầng 2: Transducer ASR
        segment_texts: list[str] = []
        for seg in speech_segments:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, seg)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                segment_texts.append(text)

        raw_acoustic_text = " ".join(segment_texts).lower().strip()
        # Chuẩn hóa Unicode NFC
        raw_acoustic_text = unicodedata.normalize("NFC", raw_acoustic_text)

        # 4. Tầng 3: XLM-RoBERTa Capu (khôi phục viết hoa + dấu câu)
        capu_text = raw_acoustic_text
        if self.enable_capu and self.capu_model is not None and raw_acoustic_text:
            try:
                capu_outputs = self.capu_model(raw_acoustic_text)
                if capu_outputs and isinstance(capu_outputs, list):
                    capu_text = capu_outputs[0].strip()
            except Exception:  # noqa: BLE001
                # Fallback nhẹ nhàng nếu câu quá ngắn hoặc lỗi model
                capu_text = raw_acoustic_text

        # 5. Tầng 4: Vietnamese ITN
        final_text = capu_text
        if self.enable_itn:
            final_text = self.itn.normalize(capu_text)

        inference_time_sec = time.perf_counter() - t0

        return ASRTranscriptionResult(
            raw_acoustic_text=raw_acoustic_text,
            capu_text=capu_text,
            final_text=final_text,
            duration_sec=duration_sec,
            inference_time_sec=inference_time_sec,
            segments_count=len(speech_segments),
            sample_rate=sr,
        )
