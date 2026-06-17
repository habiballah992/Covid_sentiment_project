import os
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


def load_whisper_model(model_size="small"):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=6
    )


def audio_file_to_text(audio_file, whisper_model, language="en"):
    if audio_file is None:
        return None

    file_name = getattr(audio_file, "name", "audio.wav")
    suffix = Path(file_name).suffix

    if suffix == "":
        suffix = ".wav"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_file.getvalue())
            temp_path = temp_audio.name

        segments, info = whisper_model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()

        if text == "":
            return None

        return text.lower().strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)