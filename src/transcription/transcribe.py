import whisper

# model_size = "tiny", "base", "small", "medium", "turbo", "large"

def transcribe_audio(audio_path, model_size="tiny", prompt=""):
    """
    Transcrit un fichier audio en texte à l’aide d’un modèle Whisper.

    Cette fonction charge un modèle Whisper préentraîné de la taille spécifiée,
    puis effectue la transcription du fichier audio fourni.  
    Elle peut également utiliser un prompt initial pour guider la transcription
    (utile pour le contexte ou la cohérence des phrases).

    Parameters
    ----------
    audio_path : str
        Chemin du fichier audio à transcrire (par ex. "tmp/record_chunk_1.wav").
    model_size : str, optional (default="tiny")
        Taille du modèle Whisper à charger.  
        Exemples valides : "tiny", "base", "small", "medium", "large".  
        Plus le modèle est grand, plus la transcription est précise mais lente.
    prompt : str, optional (default="")
        Texte initial servant de contexte au modèle pour améliorer la transcription
        (utile si le contenu audio est dans un domaine spécifique ou s’il contient des noms propres).

    Returns
    -------
    str
        Le texte transcrit à partir de l’audio.
    """
    model = whisper.load_model(model_size, initial_prompt=prompt)
    result = model.transcribe(audio_path)
    return result["text"]