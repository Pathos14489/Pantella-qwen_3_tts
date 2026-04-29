from src.logging import logging
logging.info("Importing qwen_3_tts.py...")
import src.tts_types.base_tts as base_tts
import random
import os
imported = False
try:
    logging.info("Trying to import styletts2")
    from addons.qwen_3.qwen3_tts.qwen_tts import Qwen3TTSModel
    import torch
    import soundfile as sf
    imported = True
    logging.info("Imported styletts2")
except Exception as e:
    logging.error(f"Failed to import torch and torchaudio: {e}")
    raise e
logging.info("Imported required libraries in qwen_3_tts.py")

tts_slug = "qwen_3_tts"
tts_name = "Qwen3 TTS"
default_settings = {
    "qwen_3_tts_banned_voice_models": [],
}
settings_description = {
    "qwen_3_tts_banned_voice_models": "A list of voice models to ban from being used by Qwen3TTS. This can be changed in config.json. This is useful if you have a voice model that causes issues with Qwen3TTS, such as extremely long synthesis times or crashes."
}
options = {}
settings = {}
loaded = False
description = "Qwen3TTS description goes here."
class Synthesizer(base_tts.base_Synthesizer):
    def __init__(self, conversation_manager, ttses = []):
        global tts_slug, default_settings, loaded
        super().__init__(conversation_manager)
        self.tts_slug = tts_slug
        self._default_settings = default_settings
        logging.info(f"Initializing {self.tts_slug}...")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        logging.info(f'{self.tts_slug} speaker wavs folders: {self.speaker_wavs_folders}')
        logging.config(f'{self.tts_slug} - Available voices: {self.voices()}')
        if len(self.voices()) > 0:
            random_voice = random.choice(self.voices())
            self._say("Qwen 3 T T S is ready to go.",random_voice)
        loaded = True

    def voices(self):
        """Return a list of available voices"""
        voices = super().voices()
        for banned_voice in self.config.qwen_3_tts_banned_voice_models:
            if banned_voice in voices:
                voices.remove(banned_voice)
        return voices
    
    @property
    def default_voice_model_settings(self):
        return {
            "transcription": "",
        }
    
    def _synthesize(self, voiceline, voice_model, voiceline_location, settings, aggro=0):
        """Synthesize the audio for the character specified using ParlerTTS"""
        logging.output(f'{self.tts_slug} - synthesizing {voiceline} with voice model "{voice_model}"...')
        speaker_wav_path = self.get_speaker_wav_path(voice_model)
        # settings = self.voice_model_settings(voice_model)
        logging.output(f'{self.tts_slug} - using voice model settings: {settings}')
        
        wavs, sr = self.model.generate_voice_clone(
            text=voiceline,
            language="English",
            ref_audio=speaker_wav_path,
            ref_text=settings.get("transcription", self.default_voice_model_settings["transcription"]),
        )
        sf.write(voiceline_location, wavs[0], sr)
        logging.output(f'{self.tts_slug} - synthesized {voiceline} with voice model "{voice_model}"')