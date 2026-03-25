import os
from typing import Optional

from dotenv import load_dotenv

from .azure_simulator import AzureCapacityMonitor, AzureQuotaExceededError
from .request_metrics import RequestMetricsRecorder

load_dotenv()

try:
    import azure.cognitiveservices.speech as speechsdk

    SPEECH_SDK_AVAILABLE = True
except ImportError:
    SPEECH_SDK_AVAILABLE = False


AVAILABLE_VOICES = {
    "en-US-AriaNeural": "Aria (en-US) - Female",
    "en-US-GuyNeural": "Guy (en-US) - Male",
    "en-US-JennyNeural": "Jenny (en-US) - Female",
    "en-US-SteffanNeural": "Steffan (en-US) - Male",
    "en-US-AriaMultilingualNeural": "Aria Multilingual (en-US) - Female",
    "en-US-AndrewMultilingualNeural": "Andrew Multilingual (en-US) - Male",
    "en-US-EmmaMultilingualNeural": "Emma Multilingual (en-US) - Female",
    "en-GB-SoniaNeural": "Sonia (en-GB) - Female",
    "en-GB-RyanNeural": "Ryan (en-GB) - Male",
    "de-DE-KatjaNeural": "Katja (de-DE) - Female",
    "fr-FR-DeniseNeural": "Denise (fr-FR) - Female",
    "es-ES-ElviraNeural": "Elvira (es-ES) - Female",
    "it-IT-ElsaNeural": "Elsa (it-IT) - Female",
    "jaJP-NanamiNeural": "Nanami (ja-JP) - Female",
    "koKR-SunHiNeural": "SunHi (ko-KR) - Female",
    "zhCN-XiaoxiaoNeural": "Xiaoxiao (zh-CN) - Female",
}

DEFAULT_TTS_SETTINGS = {
    "voice": "en-US-AriaNeural",
    "speed": 1.0,
    "pitch": 0,
    "style": "default",
    "volume": 100,
}


class AzureSpeechService:
    _instance: Optional["AzureSpeechService"] = None

    def __new__(cls) -> "AzureSpeechService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        if not SPEECH_SDK_AVAILABLE:
            raise RuntimeError(
                "azure-cognitiveservices-speech is not installed. "
                "Install it with: pip install azure-cognitiveservices-speech"
            )
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        if not self.speech_key or not self.speech_region:
            raise RuntimeError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in environment"
            )
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.speech_region
        )
        self.tts_settings = dict(DEFAULT_TTS_SETTINGS)
        self._apply_tts_settings()
        self.capacity_monitor = AzureCapacityMonitor()
        self._initialized = True

    def _apply_tts_settings(self) -> None:
        voice = self.tts_settings.get("voice", "en-US-AriaNeural")
        self.speech_config.speech_synthesis_voice_name = voice

    def update_tts_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.tts_settings:
                self.tts_settings[key] = value
        self._apply_tts_settings()

    def get_tts_settings(self) -> dict:
        return dict(self.tts_settings)

    def transcribe_audio(
        self,
        file_path: str,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> str:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"STT input file: {file_path}, size: {os.path.getsize(file_path)} bytes"
        )

        wav_path = file_path
        temp_dir = None
        if not file_path.lower().endswith(".wav"):
            import tempfile

            temp_dir = tempfile.mkdtemp()
            wav_path = os.path.join(temp_dir, "voice.wav")
            logger.info(f"Converting {file_path} -> {wav_path}")
            try:
                from pydub import AudioSegment

                audio = AudioSegment.from_file(file_path)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(wav_path, format="wav")
                logger.info(f"pydub conversion done: {os.path.getsize(wav_path)} bytes")
            except Exception as exc:
                logger.error(f"pydub conversion failed: {exc}")
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            file_path,
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            wav_path,
                        ],
                        capture_output=True,
                        timeout=30,
                    )
                    logger.info(
                        f"ffmpeg exit: {result.returncode}, wav size: {os.path.getsize(wav_path)}"
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"ffmpeg failed: {result.stderr.decode()[-500:]}"
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Audio conversion failed: {e}")

        logger.info(f"STT using: {wav_path} ({os.path.getsize(wav_path)} bytes)")
        duration = self.capacity_monitor.verify_stt_quota(wav_path)
        audio_config = speechsdk.AudioConfig(filename=wav_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config
        )
        result = speech_recognizer.recognize_once()
        if temp_dir and os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.capacity_monitor.register_stt_usage(duration)
            if metrics_recorder is not None:
                metrics_recorder.record_stt(
                    audio_file_path=file_path,
                    input_audio_seconds=duration,
                    transcript=result.text or "",
                )
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise ValueError("Speech not recognized. Please try again.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails.from_result(result)
            raise RuntimeError(f"Speech recognition canceled: {cancellation.reason}")
        else:
            raise RuntimeError(f"Speech recognition failed: {result.reason}")

    def synthesize_speech(
        self,
        text: str,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> bytes:
        char_count = self.capacity_monitor.verify_tts_quota(text)

        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config
        )

        voice = self.tts_settings.get("voice", "en-US-AriaNeural")
        speed = self.tts_settings.get("speed", 1.0)
        pitch = self.tts_settings.get("pitch", 0)

        rate_percent = int((speed - 1.0) * 100)
        pitch_st = f"{pitch:+d}st"

        escaped_text = self._escape_xml(text)

        ssml_text = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US' xmlns:mstts='https://www.w3.org/2001/mstts'>
  <voice name='{voice}'>
    <prosody rate='{rate_percent}%' pitch='{pitch_st}'>
      {escaped_text}
    </prosody>
  </voice>
</speak>"""

        result = speech_synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            self.capacity_monitor.register_tts_usage(char_count)
            if metrics_recorder is not None:
                metrics_recorder.record_tts(text=text, audio_bytes=result.audio_data)
            return result.audio_data
        elif result.reason == speechsdk.ResultReason.Canceled:
            error_details = getattr(result, "cancellation_details", None)
            if error_details:
                error_msg = getattr(error_details, "error_details", "Unknown")
                raise RuntimeError(f"Speech synthesis canceled: {error_msg}")
            raise RuntimeError(
                "Speech synthesis was canceled - possibly quota exceeded or invalid settings"
            )
        else:
            raise RuntimeError(f"Speech synthesis failed with reason: {result.reason}")

    @staticmethod
    def _escape_xml(text: str) -> str:
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }
        for char, escape in replacements.items():
            text = text.replace(char, escape)
        return text


_speech_singleton: Optional[AzureSpeechService] = None


def get_speech_service() -> AzureSpeechService:
    global _speech_singleton
    if _speech_singleton is None:
        _speech_singleton = AzureSpeechService()
    return _speech_singleton
