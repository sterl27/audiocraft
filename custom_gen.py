import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(description, duration=8):
    # 1. Load the model (small for speed, can be large or melody)
    print(f"Loading MusicGen model...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    
    # 2. Configure generation params
    model.set_generation_params(duration=duration)
    
    # 3. Generate audio
    print(f"Generating: '{description}' ({duration}s)...")
    wav = model.generate([description])  # generates 32kHz audio
    
    # 4. Save the output
    filename = "my_beat"
    # wav[0] contains the first (and only) generated sample
    # .cpu() moves it to CPU for saving
    audio_write(filename, wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f"Saved to {filename}.wav")

if __name__ == "__main__":
    prompt = "A chill lofi hip hop beat with a smooth jazz piano and soft drums"
    generate_music(prompt)
