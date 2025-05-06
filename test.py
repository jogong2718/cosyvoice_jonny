import sys
import os
from pathlib import Path

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

def create_audio_chunks(audio, chunk_size=16000):
    chunks = []
    for i in range(0, audio.shape[1], chunk_size):
        if i + chunk_size <= audio.shape[1]:
            chunks.append(audio[:, i:i+chunk_size])
        else:
            # Pad the last chunk if needed
            last_chunk = torch.zeros(1, chunk_size)
            last_chunk[:, :audio.shape[1]-i] = audio[:, i:]
            chunks.append(last_chunk)
    return chunks

def combine_chunks(chunks, indices):
    if not chunks or max(indices) >= len(chunks):
        return None
    return torch.cat([chunks[i] for i in indices], dim=1)

def generate_speech(model, text, audio_prompt, output_path):
    for i, result in enumerate(model.inference_cross_lingual(text, audio_prompt, stream=False)):
        torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
        break

def main():
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', 
                          load_jit=False, 
                          load_trt=False, 
                          fp16=False, 
                          use_flow_cache=False)
    
    prompt_speech_16k = load_wav('./asset/jonny_emotions.wav', 16000)
    
    audio_chunks = create_audio_chunks(prompt_speech_16k)
    print(f"Created {len(audio_chunks)} audio chunks")
    
    prompts = {
        'chinese': [
            '你好',
            '这是我的快乐。',
            '这是我悲伤',
            '这是我在大声说话',
            '这是我在窃窃私语。'
        ],
        'spanish': [
            'Hola',
            'este soy yo siendo feliz.',
            'Este soy yo triste.',
            'Este soy yo siendo muy ruidoso',
            'y este soy yo susurrando.'
        ],
        'hindi': [
            'नमस्ते',
            'यह मेरी खुशी है।',
            'यह मेरी उदासी है।',
            'यह मेरी बहुत तेज आवाज है',
            'और यह मेरी फुसफुसाहट है।'
        ],
    }
    
    chunk_combinations = [
        [0, 1],       # For prompt 0
        [0, 1],       # For prompt 1
        [2, 3, 4],    # For prompt 2
        [6, 7],       # For prompt 3
        [8, 9]        # For prompt 4
    ]

    generate_speech(
        cosyvoice,
        '你好，这是我的快乐。这是我悲伝, 这是我在大声说话，这是我在窃窃私语。',
        prompt_speech_16k,
        output_dir / 'fine_grained_control_0.wav'
    )
    
    for lang, texts in prompts.items():
        if lang == 'chinese':
            suffix = 'c'
        elif lang == 'spanish':
            suffix = 's'
        elif lang == 'hindi':
            suffix = 'h'
        
        for i, (text, chunk_indices) in enumerate(zip(texts, chunk_combinations)):
            if max(chunk_indices) < len(audio_chunks):
                audio_prompt = combine_chunks(audio_chunks, chunk_indices)
                if audio_prompt is None:
                    audio_prompt = prompt_speech_16k
            else:
                idx = min(i, len(audio_chunks) - 1)
                audio_prompt = audio_chunks[idx] if i < len(audio_chunks) else prompt_speech_16k
            
            output_file = output_dir / f'fine_grained_control_{i+1}{suffix}.wav'
            generate_speech(cosyvoice, text, audio_prompt, output_file)
            print(f"Generated {output_file}")

if __name__ == "__main__":
    main()


