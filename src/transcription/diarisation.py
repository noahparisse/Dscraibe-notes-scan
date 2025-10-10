
import torch
import whisper
import librosa
from sklearn.cluster import KMeans
import numpy as np


def diarization_with_whisper(audio_path, n_clusters=2, model_size="tiny", prompt=""):
    """
    Effectue une diarisation basique (segmentation par locuteur) à partir d'un fichier audio
    en utilisant le modèle Whisper pour la transcription et des embeddings audio pour le clustering.

    Cette fonction :
      1. Transcrit l’audio avec Whisper pour obtenir les segments de parole.
      2. Extrait les embeddings audio de chaque segment à partir du modèle Whisper.
      3. Applique un clustering K-Means pour regrouper les segments selon leur similarité vocale.
    
    L’objectif est d’obtenir une estimation des différents locuteurs dans le fichier audio.

    Parameters
    ----------
    audio_path : str
        Chemin du fichier audio à traiter (par ex. "tmp/ia_podcast.wav").
    n_clusters : int, optional (default=2)
        Nombre de locuteurs présumés (clusters K-Means).
    model_size : str, optional (default="tiny")
        Taille du modèle Whisper à utiliser.  
        Options : "tiny", "base", "small", "medium", "large".
    prompt : str, optional (default="")
        Texte de contexte initial pour améliorer la transcription (prompt initial).

    Returns
    -------
    list_start : list of float
        Liste des temps de début (en secondes) de chaque segment.
    list_end : list of float
        Liste des temps de fin (en secondes) de chaque segment.
    list_text : list of str
        Liste des transcriptions correspondantes à chaque segment.
    labels : ndarray of int
        Tableau contenant les étiquettes de cluster (locuteur estimé) pour chaque segment.
    """
    model = whisper.load_model(model_size)  

    audio, sr = librosa.load(audio_path, sr=16000)
    
    result = model.transcribe(audio_path, initial_prompt=prompt)

    list_start = []
    list_end = []
    list_text = []
    list_spectrogram = []

    for segment in result["segments"]:
        
        start_sec = segment["start"]
        end_sec = segment["end"]
        list_start.append(start_sec)
        list_end.append(end_sec)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_segment = audio[start_sample:end_sample]
        
        list_text.append(segment["text"])

   
        audio_segment = torch.from_numpy(audio_segment).float()
        audio_segment = whisper.pad_or_trim(audio_segment)  


        audio_segment = audio_segment.unsqueeze(0) 
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio_segment)
            embeddings = model.encoder(mel)
        segment_emb = embeddings.mean(dim=1).squeeze(0).numpy()  
        list_spectrogram.append(segment_emb)
        
    X = np.stack(list_spectrogram)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    return list_start, list_end, list_text, labels


    
    