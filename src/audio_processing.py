import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def load_whisper_model(model_size="small"):
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )
    return model


def audio_to_text(
    whisper_model,
    language="en",
    sample_rate=16000,
    chunk_duration=0.2,
    max_seconds=12,
    silence_seconds=1.2,
    calibration_seconds=1.0
):
    chunk_size = int(sample_rate * chunk_duration)

    print("Stay silent for calibration...")

    noise_energies = []

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    ) as stream:

        # Measure background noise
        calibration_chunks = int(calibration_seconds / chunk_duration)

        for _ in range(calibration_chunks):
            chunk, _ = stream.read(chunk_size)
            chunk = np.squeeze(chunk)

            energy = np.sqrt(np.mean(chunk ** 2))
            noise_energies.append(energy)

        noise_level = np.mean(noise_energies)
        speech_threshold = max(noise_level * 3, 0.008)

        print("Speak now...")

        audio_chunks = []
        pre_buffer = []
        started = False
        silence_time = 0
        total_time = 0

        while total_time < max_seconds:
            chunk, _ = stream.read(chunk_size)
            chunk = np.squeeze(chunk)

            energy = np.sqrt(np.mean(chunk ** 2))

            if not started:
                pre_buffer.append(chunk)

                if len(pre_buffer) > 5:
                    pre_buffer.pop(0)

            if energy > speech_threshold:
                if not started:
                    started = True
                    audio_chunks.extend(pre_buffer)

                audio_chunks.append(chunk)
                silence_time = 0

            elif started:
                audio_chunks.append(chunk)
                silence_time += chunk_duration

                if silence_time >= silence_seconds:
                    break

            total_time += chunk_duration

    if len(audio_chunks) == 0:
        print("No voice detected.")
        return None

    audio = np.concatenate(audio_chunks)

    segments, _ = whisper_model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=True
    )

    text = " ".join(segment.text.strip() for segment in segments)

    if text == "":
        return None

    return text.lower().strip()
