from denoise import denoise_audio
from diarisation import diarization
from enregistrement import record_loop
from normalize import normalize_volume
from transcribe import transcribe_w2v2
from vad import VAD
from nettoyage_audio import nettoyer_transcription_audio

from pathlib import Path
from huggingface_hub import login
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor
import os
import json
from datetime import datetime, timedelta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ctc = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
model_sample_rate = processor.feature_extractor.sampling_rate

model_segm = Model.from_pretrained(
"pyannote/segmentation-3.0")

pipeline = VoiceActivityDetection(segmentation=model_segm)


def record_loop_chunk(duration, bruit_reduction=True, samplerate=16000):
    """
    Enregistre des segments audio consécutifs et les sauvegarde dans un dossier temporaire.
    L'enregistrement s'arrête manuellement avec Ctrl+C ou automatiquement après `duration` secondes.

    Args:
        duration (float): Durée de chaque segment audio.
        bruit_reduction (bool): Appliquer une réduction de bruit.
        samplerate (int): Taux d'échantillonnage audio.
    """
    os.makedirs("tests", exist_ok=True)
    log_path = os.path.join("tests", "audio_brut.json")

    # Supprimer le fichier JSON existant
    if os.path.exists(log_path):
        os.remove(log_path)

    logs = []
    k = 1

    print("Parlez (Ctrl+C pour arrêter).")

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
            while True:
                start_time = datetime.now()
                safe_time = start_time.strftime("%Y%m%d_%H%M%S")
                filename = f"record_chunk_{k}_{safe_time}.wav"
                filepath = os.path.join("tests", filename)

                frames = []
                while (datetime.now() - start_time).total_seconds() < duration:
                    block = stream.read(1024)[0]
                    frames.append(block)

                recording = np.concatenate(frames, axis=0).squeeze()

                if bruit_reduction:
                    recording = nr.reduce_noise(y=recording, sr=samplerate)

                sf.write(filepath, recording, samplerate)
                end_time = datetime.now()

                entry = {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": filename
                }
                logs.append(entry)

                # Sauvegarde du JSON après chaque segment
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=4, ensure_ascii=False)

                k += 1

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C).")

        # Sauvegarde du dernier segment partiel (si existant)
        if 'frames' in locals() and len(frames) > 0:
            recording = np.concatenate(frames, axis=0).squeeze()
            if bruit_reduction:
                recording = nr.reduce_noise(y=recording, sr=samplerate)
            sf.write(filepath, recording, samplerate)
            end_time = datetime.now()

            entry = {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename
            }
            logs.append(entry)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)


def VADe(audio_path, min_duration_on=2.0, min_duration_off=2.0):

    HYPER_PARAMETERS = {
    # Si un segment de parole détecté dure moins de 3 secondes, il sera ignoré.
    "min_duration_on": min_duration_on,
    # Si une pause est plus courte que 10 secondes, elle peut être remplie ou fusionnée avec les segments voisins.
    "min_duration_off": min_duration_off
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_path)

    log_path = os.path.join("tmp", "transcriptions_log.json")

    # Charger le JSON existant s'il existe
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
        
        
    #Time
    filename_brut = audio_path.name

    with open("tests/audio_brut.json", "r") as f:
        data = json.load(f)

    entry = next((item for item in data if item["filename"] == filename_brut ), None)

    if entry:
        i=0
    else:
        print("Aucun enregistrement trouvé pour ce fichier.")
    
    
    start_time = entry["start_time"]

    
    waveform, sr = torchaudio.load(audio_path)
    i=0
    for segment, _, _ in vad.itertracks(yield_label=True):
        print(segment)
        i = i+1
        
        s = int(segment.start)
        e = int(segment.end)
        
        start = int(segment.start * sr)
        end = int(segment.end * sr)
        
        segment = waveform[:, start:end]
        
        segment_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{i}.wav"
        segment_path = os.path.join("tmp", segment_filename)
        
        torchaudio.save(segment_path, segment, sr)
        
        
        base_start_str = start_time
        base_start = datetime.strptime(base_start_str, "%Y-%m-%d %H:%M:%S")


        absolute_start = base_start + timedelta(seconds=s)
        absolute_end = base_start + timedelta(seconds=e)
        
        
        entry = {
            "start_time": absolute_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": absolute_end.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": segment_filename
        }
        logs.append(entry)
        
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
        
    # --- AJOUT DU TRI PAR DATE APRÈS SAUVEGARDE ---
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    logs.sort(key=lambda x: datetime.strptime(x["start_time"], "%Y-%m-%d %H:%M:%S"))


    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
    # ------------------------------------------------ #

    
    data = [entry for entry in data if entry.get("filename") != filename_brut]
    with open("tests/audio_brut.json", "w") as f:
        json.dump(data, f, indent=4)
    
    os.remove(audio_path)
    


def transcribe_w2v2_clean(audio_path):
    """
    Transcrit un fichier audio en texte à l’aide d’un modèle Whisper.

    Cette fonction charge un modèle Whisper préentraîné de la taille spécifiée,
    puis effectue la transcription du fichier audio fourni.  
    Elle peut également utiliser un prompt initial pour guider la transcription

    """
    
    log_path = os.path.join("tmp", "transcriptions_log.json")

    # Charger le JSON existant s'il existe
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
        
        
    #Time
    filename_brut = audio_path.name

    entry = next((item for item in logs if item["filename"] == filename_brut ), None)

    
    
    wav_path = audio_path
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.squeeze(axis=0)  # mono

    # resample
    if sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
        waveform = resampler(waveform)

    # normalize
    input_dict = processor(waveform, sampling_rate=model_sample_rate, return_tensors="pt")

    with torch.inference_mode():
        logits = model_ctc(input_dict.input_values.to(device)).logits

    # decode
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)[0]

    entry["transcription"] = predicted_sentence
    
    entry["transcription_clean"] = nettoyer_transcription_audio(predicted_sentence)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
