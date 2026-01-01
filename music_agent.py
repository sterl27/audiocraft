import os
import torch
import torchaudio
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load API keys from .env
load_dotenv()

class MusicAgent:
    def __init__(self, model_name='facebook/musicgen-small'):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.eleven_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        print(f"Initializing MusicGen ({model_name})...")
        self.music_model = MusicGen.get_pretrained(model_name)

    def generate_musical_prompt(self, user_intent):
        """Use GPT to turn a simple idea into a rich MusicGen description."""
        print(f"Consulting OpenAI for: {user_intent}...")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a musical prompt engineer. Convert user requests into detailed descriptions for an AI music generator (MusicGen). Include style, instruments, tempo, and mood. Keep it under 200 characters."},
                {"role": "user", "content": user_intent}
            ]
        )
        return response.choices[0].message.content

    def generate_voiceover(self, text, filename="intro_voice"):
        """Generate a voiceover using ElevenLabs."""
        print(f"Generating ElevenLabs voiceover: '{text}'...")
        audio = self.eleven_client.generate(
            text=text,
            voice="Rachel",  # Or any other voice ID/name
            model="eleven_multilingual_v2"
        )
        save(audio, f"{filename}.mp3")
        return f"{filename}.mp3"

    def create_track(self, intent, duration=10, voice_text=None):
        """The full pipeline: GPT-prompt -> MusicGen -> ElevenLabs."""
        # 1. Expand prompt with GPT
        prompt = self.generate_musical_prompt(intent)
        print(f"Final Prompt: {prompt}")

        # 2. Generate Music
        self.music_model.set_generation_params(duration=duration)
        print(f"Generating music...")
        wav = self.music_model.generate([prompt])
        
        music_filename = "agent_output"
        audio_write(music_filename, wav[0].cpu(), self.music_model.sample_rate, strategy="loudness", add_suffix=True)
        
        # 3. Handle Voiceover if requested
        if voice_text:
            voice_path = self.generate_voiceover(voice_text)
            print(f"Voiceover saved to {voice_path}")

        print(f"Success! Music saved to {music_filename}.wav")

if __name__ == "__main__":
    # Example usage
    agent = MusicAgent()
    
    intent = "A high-energy cyberpunk chase scene"
    intro_text = "Buckle up. We're heading into the neon district."
    
    agent.create_track(intent, duration=8, voice_text=intro_text)
